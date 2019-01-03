from __future__ import print_function
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
import itertools
import subprocess
import os

eps = 1e-15


def TD_MVDR(mixture, interference, reference=None, frame_len=512, frame_step=1):
    """ Time Domain Minimum Variance Distortionless Response (MVDR) Beamformer as described in
    https://ieeexplore.ieee.org/xpl/ebooks/bookPdfWithBanner.jsp?fileName=6504598.pdf&bkn=6497230&pdfType=chapter
    Like all the other beamformers in the library the td_mvdr receives as input the multichannel recording of the
    mixture (rec), of the target speaker (ref) and of the interference (noise).
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param interference: nd_array (n_mics, time) of the noise recordings
    :param reference: nd_array (n_mics, time) of the reference recordings
    :param frame_len:
    :param frame_step:
    :return: the enhanced signal

    Note: It is highly recommended NOT to change the frame_step. A frame step higher then 1 will be faster to calculate
    but will not be as effective.
    """

    M, T = mixture.shape  # (n_mics, time)
    n_win = (T - frame_len) // frame_step  # number of windows

    R_y = 0.0
    rho_yY = 0.0
    rho_vV = 0.0
    rho_xX = 0.0 if reference is not None else None

    var_y = np.var(mixture[0])
    var_v = np.var(interference[0])

    for i in range(n_win):
        j = i * frame_step

        # pick frame
        v = interference[:, j:j + frame_len]  # interference
        y = mixture[:, j:j + frame_len]  # mixture

        # vectorize frame
        Y = y.reshape(-1)
        V = v.reshape(-1)

        # mixture covariance matrix
        R_y += np.outer(Y, Y)

        # steering vector (depending if reference is available)
        if reference is not None:  # with real ref
            x = reference[:, j:j + frame_len]
            X = x.reshape(-1)
            rho_xX += x[0][-1] * X
        else:  # with subtraction
            rho_yY += y[0][-1] * Y
            rho_vV += v[0][-1] * V

    # calculate weights with collected statistics
    rho_xX = rho_xX / n_win if rho_xX is not None else (rho_yY - rho_vV) / (var_y - var_v) / n_win

    # calculate weights
    part = np.linalg.inv(R_y / n_win + np.eye(M * frame_len) * 1e-15).dot(rho_xX)
    h_MVDR = part / rho_xX.T.dot(part) 
    h_MVDR /= np.sqrt(np.sum(h_MVDR ** 2))

    # apply weights frame by frame
    y_MVDR = np.zeros((T,))
    for t in range(T - frame_len):
        y = mixture[:, t:t + frame_len].reshape(-1)
        y_MVDR[frame_len + t] = h_MVDR.dot(y)

    # normalize
    y_MVDR /= np.max(np.abs(y_MVDR))
    
    return y_MVDR


def TD_MWF(mixture, interference, reference=None, frame_len=512, frame_step=1):
    """ Time Domain Multichannel Wiener Filter (MWF) Beamformer as described in
    https://ieeexplore.ieee.org/xpl/ebooks/bookPdfWithBanner.jsp?fileName=6504598.pdf&bkn=6497230&pdfType=chapter
    Like all the other beamformers in the library the td_mwf receives as input the multichannel recording of the
    mixture (mixture), of the target speaker (reference) and of the interference (interference). And produces a single
    channel enhanced signal of the desired speaker.
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param interference: nd_array (n_mics, time) of the noise recordings
    :param reference: nd_array (n_mics, time) of the reference recordings
    :param frame_len:
    :param frame_step:
    :return: the enhanced signal

    Note: It is highly recommended not to change the frame_step. A frame step higher then 1 will be faster to calculate
    but will not be as effective.
    """
    M, T = mixture.shape  # (n_mics, time)
    n_win = (T - frame_len) // frame_step  # number of windows

    R_in = 0.0
    rho_yY = 0.0
    rho_vV = 0.0
    rho_xX = 0.0 if reference is not None else None

    var_y = np.var(mixture[0])
    var_v = np.var(interference[0])
    var_x = np.var(reference[0]) if reference is not None else np.var(mixture[0])

    for i in range(n_win):
        j = i * frame_step

        # pick frame
        v = interference[:, j:j + frame_len]  # interference
        y = mixture[:, j:j + frame_len]  # mixture

        # vectorize frame
        Y = y.reshape(-1)
        V = v.reshape(-1)

        # mixture covariance matrix
        R_in += np.outer(V, V)

        # steering vector (depending if reference is available)
        if reference is not None:  # with real ref
            x = reference[:, j:j + frame_len]
            X = x.reshape(-1)
            rho_xX += x[0][-1] * X
        else:  # with subtraction
            rho_yY += y[0][-1] * Y
            rho_vV += v[0][-1] * V

    # calculate weights with collected statistics
    rho_xX = rho_xX / n_win if rho_xX is not None else (rho_yY - rho_vV) / (var_y - var_v) / n_win

    # R_y - inv
    part = np.linalg.inv(R_in / n_win + np.eye(M * frame_len) * 1e-15).dot(rho_xX)  # / np.linalg.norm(rho_xX)
    h_MWF = (part * var_x) / (1 + var_x * rho_xX.T.dot(part))
    h_MWF /= np.sqrt(np.sum(h_MWF ** 2))

    y_MWF = np.zeros((T,))

    for t in range(T - frame_len):
        y = mixture[:, t:t + frame_len].reshape(-1)
        y_MWF[frame_len + t] = h_MWF.dot(y)

    y_MWF /= np.max(np.abs(y_MWF))

    return y_MWF


def MVDR(mixture, interference, reference, frame_len=2048, frame_step=512):
    """
    Frequency domain Minimum Variance Distortionless Response (MVDR) beamformer
    :param mixture:
    :param interference:
    :param reference:
    :param frame_len:
    :param frame_step:
    :return:
    """
    # calculate stft
    mixture_stft = stft(mixture, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]

    # estimate steering vector for desired speaker (depending if reference is available)
    if reference is not None:
        reference_stft = stft(reference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
        h = estimate_steering_vector(reference_stft=reference_stft)
    else:
        noise_spec = stft(interference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
        h = estimate_steering_vector(mixture_stft=mixture_stft, interference_stft=noise_spec)

    # calculate weights
    w = mvdr_weights(mixture_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)
    
    # reconstruct wav
    recon = istft(sep_spec, noverlap=(frame_len - frame_step))[1]
    
    return recon


def MSNR(mixture, interference, reference=None, frame_len=2048, frame_step=512):

    # calculate stft
    mixture_stft = stft(mixture, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
    interference_stft = stft(interference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]

    # estimate steering vector for desired speaker (depending if reference is available)
    if reference is not None:
        reference_stft = stft(reference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
        h = estimate_steering_vector(reference_stft=reference_stft)
    else:
        h = estimate_steering_vector(mixture_stft=mixture_stft, interference_stft=interference_stft)

    # calculate weights
    w = mvdr_weights(interference_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    reconstructed = istft(sep_spec, noverlap=(frame_len - frame_step))[1]

    return reconstructed

    
def estimate_steering_vector(reference_stft=None, mixture_stft=None, interference_stft=None):
    """
    Estimation of steering vector based on microphone recordings. The eigenvector technique used is described in
    Sarradj, E. (2010). A fast signal subspace approach for the determination of absolute levels from phased microphone
    array measurements. Journal of Sound and Vibration, 329(9), 1553-1569.
    The steering vector is represented by the leading eigenvector of the covariance matrix calculated for each
    frequency separately.
    :param reference_stft: nd_array (channels, time, freq_bins)
    :param mixture_stft: nd_array (channels, time, freq_bins)
    :param interference_stft: nd_array (channels, time, freq_bins)
    :return: h: nd_array (freq_bins, ): steering vector
    """

    if reference_stft is None:
        if mixture_stft is None or interference_stft is None:
            raise ValueError("If no reference recordings are provided you need to provide both mixture recordings "
                             "and noise recordings")
        C, F, T = mixture_stft.shape  # (channels, freq_bins, time)
    else:
        C, F, T = reference_stft.shape  # (channels, freq_bins, time)

    eigen_vec, eigen_val, h = [], [], []

    for f in range(F):  # Each frequency separately

        # covariance matrix
        if reference_stft is None:
            # covariance matrix estimated by subtracting mixture and noise covariances
            _R0 = mixture_stft[:, f].dot(np.conj(mixture_stft[:, f].T))
            _R1 = interference_stft[:, f].dot(np.conj(interference_stft[:, f].T))
            _Rxx = _R0 - _R1
        else:
            # covariance matrix estimated directly from single speaker
            _Rxx = reference_stft[:, f].dot(np.conj(reference_stft[:, f].T))

        # eigendecomposition
        [_d, _v] = np.linalg.eig(_Rxx)

        # index of leading eigenvector
        idx = np.argsort(_d)[::-1][0]

        # collect leading eigenvector and eigenvalue
        eigen_val.append(_d[idx])
        eigen_vec.append(_v[:, idx])

    # rescale eigenvectors by eigenvalues for each frequency
    for vec, val in zip(eigen_vec, eigen_val):
        h.append(vec * val / np.abs(val))

    # return steering vector
    return np.vstack(h)


def mvdr_weights(mixture_stft, h):

    C, F, T = mixture_stft.shape  # (channels, freq_bins, time)

    # covariance matrix
    R_y = np.einsum('a...c,b...c', mixture_stft, np.conj(mixture_stft)) / T

    # preallocate weights
    W = np.zeros((F, C), dtype='complex64')

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):

        part = np.linalg.inv(r + np.eye(C, dtype='complex') * eps).dot(_h)
        _w = part / np.conj(_h).T.dot(part)

        W[i, :] = _w

    return W


def apply_beamforming_weights(signals, weights):
    """
    Fastest way to apply beamforming weights in frequency domain.
    :param signals: nd_array (freq_bins (a), n_mics (b))
    :param weights: nd_array (n_mics (b), freq_bins (a), time_frames (c))
    :return: nd_array (freq_bins (a), time_frames (c)): filtered stft
    """
    return np.einsum('ab,bac->ac', np.conj(weights), signals)


def sdw_mwf_weights(reference_mic, interference_stft, h, mu):

    C, F, T = interference_stft.shape  # (channels, freq_bins, time)
    Tss = np.mean(np.abs(reference_mic) ** 2)

    # covariance matrix
    R_y = np.einsum('a...c,b...c', interference_stft, np.conj(interference_stft)) / T  # (freq_bins, channels, channels)

    # preallocate weights
    W = np.zeros((F, C), dtype='complex64')

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):
        part = Tss * np.linalg.inv(r + np.eye(C) * eps).dot(_h)
        _w = part / (mu + Tss * np.conj(_h).T.dot(part))
        W[i, :] = _w

    return W


def SDW_MWF(mixture, interference, reference=None, mu=0, frame_len=2048, frame_step=512):
    
    mixture_stft = stft(mixture, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
    interference_stft = stft(interference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]

    if reference is not None:
        reference_stft = stft(reference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
        # estimate forward mapping for desired speaker
        h = estimate_steering_vector(reference_stft=reference_stft)
    else:
        # estimate forward mapping for desired speaker
        h = estimate_steering_vector(mixture_stft=mixture_stft, interference_stft=interference_stft)

    # calculate weights
    w = sdw_mwf_weights(mixture[0], interference_stft, h, mu)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)
    
    # reconstruct wav
    recon = istft(sep_spec, noverlap=(frame_len - frame_step))[1]

    # recon = np.concatenate([recon, np.zeros((len(mixture[0]) - len(recon),))])
    
    return recon


def invert(M):
    """
    This function inverts the last 2 dimensions of the nd_array input
    :param M: nd_array: any dimension but the last 2 should be equal for inversion
    :return: nd_array of inverted matrices
    """

    if M.shape[-2] != M.shape[-1]:
        raise ValueError("Should be a square matrix or array of square matrices")
    if len(M.shape) == 2:
        return np.linalg.inv(M + np.eye(M.shape[0], dtype=M.dtype) * eps)
    else:
        return np.array([invert(m) for m in M])


def compute_psd(x_stft):

    C, F, T = x_stft.shape

    # Learn Power Spectral Density and spatial covariance matrix
    Rjj = np.einsum('abc,dbc->bcad', x_stft, np.conj(x_stft))

    # 2/ compute first naive estimate of the source spectrogram as the
    #    average of spectrogram over channels
    P = np.mean(np.abs(x_stft) ** 2, axis=0)

    # 3/ take the spatial covariance matrix as the average of
    #    the observed Rjj weighted Rjj by 1/Pj. This is because the
    #    covariance is modeled as Pj Rj
    R = np.mean(Rjj / (eps + P[..., None, None]), axis=1)

    # add some regularization to this estimate: normalize and add small
    # identify matrix, so we are sure it behaves well numerically.
    R = R * C / np.trace(R) + eps * np.tile(
        np.eye(C, dtype='complex64')[None, ...], (F, 1, 1)
    )

    # 4/ Now refine the power spectral density estimate. This is to better
    #    estimate the PSD in case the source has some correlations between
    #    channels.

    #    invert Rj
    Rj_inv = invert(R)

    #    now compute the PSD
    P = 0
    for (i1, i2) in itertools.product(range(C), range(C)):
        P += 1. / C * np.real(
            Rj_inv[:, i1, i2][:, None] * Rjj[..., i2, i1]
        )

    return P, R


def MWF_Oracle(mixture, interference, reference, frame_len=2048, frame_step=512):

    N = mixture.shape[1]

    # compute stft
    mixture_stft = stft(mixture, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
    interference_stft = stft(interference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]
    reference_stft = stft(reference, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]

    # compute PSD for reference and interference
    P_reference, R_reference = compute_psd(reference_stft)
    P_interference, R_interference = compute_psd(interference_stft)

    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    cov_reference = P_reference[..., None, None] * R_reference[:, None, ...]
    cov_interference = P_interference[..., None, None] * R_interference[:, None, ...]
    Cxx = cov_reference + cov_interference

    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx)

    # computes multichannel Wiener gain as Pj Rj invCxx
    G = np.einsum('abcd,abde->abce', cov_reference, invCxx)

    # separates by (matrix-)multiplying this gain with the mix.
    filtered_stft = np.einsum('abdc,cab->dab', G, mixture_stft)

    # invert to time domain
    reconstructed = istft(filtered_stft, noverlap=(frame_len - frame_step))[1][0, :N]
        
    return reconstructed


def BeamformIt(mixture, fs=8000, basedir='/Data/software/BeamformIt/', verbose=False):

    mixture /= np.max(np.abs(mixture))

    if not os.path.exists('/tmp/audios/'):
        os.mkdir('/tmp/audios/')

    wavfile.write('/tmp/audios/rec.wav', fs, mixture.T)

    p = subprocess.Popen("cd {}; bash do_beamforming.sh /tmp/audios/ temps".format(basedir), stdout=subprocess.PIPE, shell=True)
    
    (output, err) = p.communicate()
    p_status = p.wait()
    if verbose:
        print("Output: {}".format(output))
        print("Error: {}".format(err))
        print("Status: {}".format(p_status))

    _, s = wavfile.read('{}/output/temps/temps.wav'.format(basedir))
    s = s.astype('float32')
    s /= np.max(np.abs(s))
    return s
