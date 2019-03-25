from __future__ import print_function
import numpy as np
from scipy.io import wavfile
from scipy.linalg import solve, eigh, LinAlgError
from scipy.signal import stft as _stft, istft as _istft
import itertools
import subprocess
import os
import soundfile as sf

eps = 1e-15


def stft(x, frame_len=2048, frame_step=512):
    return _stft(x, nperseg=frame_len, noverlap=(frame_len - frame_step))[-1]


def istft(x, frame_len=2048, frame_step=512, input_len=None):
    _reconstructed = _istft(x, noverlap=(frame_len - frame_step))[1].astype('float32' if x.dtype == 'complex64' else 'float64')
    if input_len is None:
        return _reconstructed
    else:
        rec_len = len(_reconstructed)
        if input_len <= rec_len:
            return _reconstructed[:input_len]
        else:
            return np.append(_reconstructed, np.zeros((input_len - rec_len,), dtype=x.dtype))


def TD_MVDR(mixture, noise, target=None, frame_len=512, frame_step=1):
    """ Time Domain Minimum Variance Distortionless Response (MVDR) Beamformer as described in
    https://ieeexplore.ieee.org/xpl/ebooks/bookPdfWithBanner.jsp?fileName=6504598.pdf&bkn=6497230&pdfType=chapter
    Like all the other beamformers in the library the td_mvdr receives as input the multichannel recording of the
    mixture (rec), of the target speaker (ref) and of the noise (noise).
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
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
    rho_xX = 0.0 if target is not None else None

    var_y = np.var(mixture[0])
    var_v = np.var(noise[0])

    for i in range(n_win):
        j = i * frame_step

        # pick frame
        v = noise[:, j:j + frame_len]  # noise
        y = mixture[:, j:j + frame_len]  # mixture

        # vectorize frame
        Y = y.reshape(-1)
        V = v.reshape(-1)

        # mixture covariance matrix
        R_y += np.outer(Y, Y)

        # steering vector (depending if target is available)
        if target is not None:  # with real ref
            x = target[:, j:j + frame_len]
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


def TD_MWF(mixture, noise, target=None, frame_len=512, frame_step=1):
    """ Time Domain Multichannel Wiener Filter (MWF) Beamformer as described in
    https://ieeexplore.ieee.org/xpl/ebooks/bookPdfWithBanner.jsp?fileName=6504598.pdf&bkn=6497230&pdfType=chapter
    Like all the other beamformers in the library the td_mwf receives as input the multichannel recording of the
    mixture (mixture), of the target speaker (target) and of the noise (noise). And produces a single
    channel enhanced signal of the desired speaker.
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
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
    rho_xX = 0.0 if target is not None else None

    var_y = np.var(mixture[0])
    var_v = np.var(noise[0])
    var_x = np.var(target[0]) if target is not None else np.var(mixture[0])

    for i in range(n_win):
        j = i * frame_step

        # pick frame
        v = noise[:, j:j + frame_len]  # noise
        y = mixture[:, j:j + frame_len]  # mixture

        # vectorize frame
        Y = y.reshape(-1)
        V = v.reshape(-1)

        # mixture covariance matrix
        R_in += np.outer(V, V)

        # steering vector (depending if target is available)
        if target is not None:  # with real ref
            x = target[:, j:j + frame_len]
            X = x.reshape(-1)
            rho_xX += x[0][-1] * X
        else:  # with subtraction
            rho_yY += y[0][-1] * Y
            rho_vV += v[0][-1] * V

    # calculate weights with collected statistics
    rho_xX = rho_xX / n_win if rho_xX is not None else (rho_yY - rho_vV) / (var_y - var_v) / n_win

    # R_y - inv
    part = np.linalg.inv(R_in / n_win + np.eye(M * frame_len) * 1e-15).dot(rho_xX)
    h_MWF = (part * var_x) / (1 + var_x * rho_xX.T.dot(part))
    h_MWF /= np.sqrt(np.sum(h_MWF ** 2))

    y_MWF = np.zeros((T,))

    for t in range(T - frame_len):
        y = mixture[:, t:t + frame_len].reshape(-1)
        y_MWF[frame_len + t] = h_MWF.dot(y)

    y_MWF /= np.max(np.abs(y_MWF))

    return y_MWF


def MVDR(mixture, noise, target=None, frame_len=2048, frame_step=512, ref_mic=0):
    """
    ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf
    Frequency domain Minimum Variance Distortionless Response (MVDR) beamformer
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :return: the enhanced signal
    """
    # calculate stft
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)

    # estimate steering vector for desired speaker (depending if target is available)
    if target is not None:
        target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(target_stft=target_stft)
    else:
        noise_spec = stft(noise, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_spec)

    # calculate weights
    w = mvdr_weights(mixture_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    recon = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))

    return recon


def MSNR(mixture, noise, target=None, frame_len=2048, frame_step=512, ref_mic=0):
    """
    Frequency domain Maximum Signal-to-Noise Ratio (MSNR) beamformer, following the formulation in
    ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf.
    :param mixture:
    :param noise:
    :param target:
    :param frame_len:
    :param frame_step:
    :return:
    """
    # ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf
    # calculate stft
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    # estimate steering vector for desired speaker (depending if target is available)
    if target is not None:
        target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
        h = estimate_steering_vector(target_stft=target_stft)
    else:
        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_stft)

    # calculate weights
    w = mvdr_weights(noise_stft, h)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    reconstructed = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))

    return reconstructed


def estimate_steering_vector(target_stft=None, mixture_stft=None, noise_stft=None):
    """
    Estimation of steering vector based on microphone recordings. The eigenvector technique used is described in
    Sarradj, E. (2010). A fast signal subspace approach for the determination of absolute levels from phased microphone
    array measurements. Journal of Sound and Vibration, 329(9), 1553-1569.
    The steering vector is represented by the leading eigenvector of the covariance matrix calculated for each
    frequency separately.
    :param target_stft: nd_array (channels, time, freq_bins)
    :param mixture_stft: nd_array (channels, time, freq_bins)
    :param noise_stft: nd_array (channels, time, freq_bins)
    :return: h: nd_array (freq_bins, ): steering vector
    """

    if target_stft is None:
        if mixture_stft is None or noise_stft is None:
            raise ValueError("If no target recordings are provided you need to provide both mixture recordings "
                             "and noise recordings")
        C, F, T = mixture_stft.shape  # (channels, freq_bins, time)
    else:
        C, F, T = target_stft.shape  # (channels, freq_bins, time)

    eigen_vec, eigen_val, h = [], [], []

    for f in range(F):  # Each frequency separately

        # covariance matrix
        if target_stft is None:
            # covariance matrix estimated by subtracting mixture and noise covariances
            _R0 = mixture_stft[:, f].dot(np.conj(mixture_stft[:, f].T))
            _R1 = noise_stft[:, f].dot(np.conj(noise_stft[:, f].T))
            _Rxx = _R0 - _R1
        else:
            # covariance matrix estimated directly from single speaker
            _Rxx = target_stft[:, f].dot(np.conj(target_stft[:, f].T))

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
    R_y = condition_covariance(R_y, 1e-6)
    R_y /= np.trace(R_y, axis1=-2, axis2=-1)[..., None, None]
    # preallocate weights
    W = np.zeros((F, C), dtype='complex64')

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):
        # part = np.linalg.inv(r + np.eye(C, dtype='complex') * eps).dot(_h)
        part = solve(r, _h)
        _w = part / np.conj(_h).T.dot(part)

        W[i, :] = _w

    return W


def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x, axis1=-2, axis2=-1)[..., None, None] / x.shape[-1]
    n = len(x.shape) - 2
    scaled_eye = np.eye(x.shape[-1], dtype=x.dtype)[(None,) * n] * scale
    return (x + scaled_eye) / (1 + gamma)


def apply_beamforming_weights(signals, weights):
    """
    Fastest way to apply beamforming weights in frequency domain.
    :param signals: nd_array (freq_bins (a), n_mics (b))
    :param weights: nd_array (n_mics (b), freq_bins (a), time_frames (c))
    :return: nd_array (freq_bins (a), time_frames (c)): filtered stft
    """
    return np.einsum('ab,bac->ac', np.conj(weights), signals)


def sdw_mwf_weights(target_mic, noise_stft, h, mu):
    C, F, T = noise_stft.shape  # (channels, freq_bins, time)
    Tss = np.mean(np.abs(target_mic) ** 2)

    # covariance matrix
    R_y = np.einsum('a...c,b...c', noise_stft, np.conj(noise_stft)) / T  # (freq_bins, channels, channels)
    R_y = condition_covariance(R_y, 1e-6)
    R_y /= np.trace(R_y, axis1=-2, axis2=-1)[..., None, None]

    # preallocate weights
    W = np.zeros((F, C), dtype='complex64')

    # compute weights for each frequency separately
    for i, r, _h in zip(range(F), R_y, h):
        # part = Tss * np.linalg.inv(r + np.eye(C) * eps).dot(_h)
        part = Tss * solve(r, _h)
        _w = part / (mu + Tss * np.conj(_h).T.dot(part)) * np.conj(_h[0])
        W[i, :] = _w

    return W


def SDW_MWF(mixture, noise, target=None, mu=0, frame_len=2048, frame_step=512, ref_mic=0):
    """
    Speech Distortion Weighted Multi-channel Wiener Filter (SDW_MWF) following the formulation in
    ftp://ftp.esat.kuleuven.ac.be/stadius/spriet/reports/08-211.pdf.
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param mu: float, the allowed speech distortion parameter (0 == distortionless)
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :return: the enhanced signal
    """
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    if target is not None:
        target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
        # estimate forward mapping for desired speaker
        h = estimate_steering_vector(target_stft=target_stft)
    else:
        # estimate forward mapping for desired speaker
        h = estimate_steering_vector(mixture_stft=mixture_stft, noise_stft=noise_stft)

    # calculate weights
    w = sdw_mwf_weights(mixture[0], noise_stft, h, mu)

    # apply weights
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    # reconstruct wav
    recon = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))

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


def MWF_Oracle(mixture, noise, target, frame_len=2048, frame_step=512):
    N = mixture.shape[1]

    # compute stft
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)
    target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)

    # compute PSD for target and noise
    P_target, R_target = compute_psd(target_stft)
    P_noise, R_noise = compute_psd(noise_stft)

    # All parameters are estimated. compute the mix covariance matrix as
    # the sum of the sources covariances.
    cov_target = P_target[..., None, None] * R_target[:, None, ...]
    cov_noise = P_noise[..., None, None] * R_noise[:, None, ...]
    Cxx = cov_target + cov_noise

    # we need its inverse for computing the Wiener filter
    invCxx = invert(Cxx)

    # computes multichannel Wiener gain as Pj Rj invCxx
    G = np.einsum('abcd,abde->abce', cov_target, invCxx)

    # separates by (matrix-)multiplying this gain with the mix.
    filtered_stft = np.einsum('abdc,cab->dab', G, mixture_stft)[0]

    # invert to time domain
    reconstructed = istft(filtered_stft, frame_len=frame_len, frame_step=frame_step, input_len=N)

    return reconstructed


def BeamformIt(mixture, fs=8000, basedir='/Data/software/BeamformIt/', verbose=False):
    mixture /= np.max(np.abs(mixture))

    if not os.path.exists('/tmp/audios/'):
        os.mkdir('/tmp/audios/')

    wavfile.write('/tmp/audios/rec.wav', fs, mixture.T)

    p = subprocess.Popen("cd {}; bash do_beamforming.sh /tmp/audios/ temps".format(basedir),
                         stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()
    p_status = p.wait()
    if verbose:
        print("Output: {}".format(output))
        print("Error: {}".format(err))
        print("Status: {}".format(p_status))

    s, _ = sf.read('{}/output/temps/temps.wav'.format(basedir))

    return s


def mb_mvdr_weights(mixture_stft, mask_noise, mask_target, phase_correct=False):
    """
    Calculates the MB MVDR weights in frequency domain
    :param mixture_stft: nd_array (channels, freq_bins, time)
    :param mask_noise: 2d array (freq_bins, time)
    :param mask_target: 2d array (freq_bins, time)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :return: the gev weights: 2d array (freq_bins, channels)
    """
    C, F, T = mixture_stft.shape  # (channels, freq_bins, time)

    # covariance matrices
    cov_noise = get_power_spectral_density_matrix(mixture_stft.transpose(1, 0, 2), mask_noise, normalize=False)
    cov_speech = get_power_spectral_density_matrix(mixture_stft.transpose(1, 0, 2), mask_target, normalize=True)
    cov_noise = condition_covariance(cov_noise, 1e-6)
    cov_noise /= np.trace(cov_noise, axis1=-2, axis2=-1)[..., None, None]

    h = []
    for f in range(F):
        try:
            _cov_noise = cov_noise[f]
            _cov_speech = cov_speech[f]

            # mask-based MVDR
            _G = solve(_cov_noise, _cov_speech)
            _lambda = np.trace(_G)
            h_r = (1. / _lambda) * _G[:, 0]
            h.append(h_r)

        except LinAlgError:  # just a precaution if the solve does not work
            h.append(np.ones((C,)) + 1j * np.ones((C,)))

    w = np.array(h)
    if phase_correct:
        w = phase_correction(w)
    return w


def MB_MVDR(mixture, noise, target, mask="IBM", frame_len=2048, frame_step=512, phase_correct=False,
                   ref_mic=0):
    """
    Mask based MVDR beamformer as formulated in http://www.jonathanleroux.org/pdf/Erdogan2016Interspeech09.pdf.
    This implementation uses oracle masks but you can use mb_mvdr_weights to get the weights if you have custom masks
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param mask: type of oracle mask: IBM, IRM, WFM, or PSM (see https://arxiv.org/pdf/1709.00917.pdf and
    https://arxiv.org/pdf/1809.07454.pdf)
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :param ref_mic: int, (self explanatory)
    :return: the enhanced signal
    """

    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    mask_target, mask_noise = calculate_masks([target_stft, noise_stft], mask=mask)

    w = mb_mvdr_weights(target_stft + noise_stft, mask_noise[ref_mic], mask_target[ref_mic], phase_correct=phase_correct)
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    reconstructed = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))
    return reconstructed


def MB_MVDR_oracle(mixture, noise, target, mask="IBM", frame_len=2048, frame_step=512, phase_correct=False,
                   ref_mic=0):
    """
    Mask based MVDR beamformer as formulated in http://www.jonathanleroux.org/pdf/Erdogan2016Interspeech09.pdf.
    This implementation uses oracle masks but you can use mb_mvdr_weights to get the weights if you have custom masks
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param mask: type of oracle mask: IBM, IRM, WFM, or PSM (see https://arxiv.org/pdf/1709.00917.pdf and
    https://arxiv.org/pdf/1809.07454.pdf)
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :param ref_mic: int, (self explanatory)
    :return: the enhanced signal
    """

    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    mask_target, mask_noise = calculate_masks([target_stft, noise_stft], mask=mask)

    w = mb_mvdr_weights(mixture_stft, mask_noise[ref_mic], mask_target[ref_mic], phase_correct=phase_correct)
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    reconstructed = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))
    return reconstructed


def mb_gev_weights(mixture_stft, mask_noise, mask_target, phase_correct=False):
    """
    Calculates the MB GEV weights in frequency domain
    :param mixture_stft: nd_array (channels, freq_bins, time)
    :param mask_noise: 2d array (freq_bins, time)
    :param mask_target: 2d array (freq_bins, time)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :return: the gev weights: 2d array (freq_bins, channels)
    """
    C, F, T = mixture_stft.shape  # (channels, freq_bins, time)

    # covariance matrices
    cov_noise = get_power_spectral_density_matrix(mixture_stft.transpose(1, 0, 2), mask_noise, normalize=False)
    cov_speech = get_power_spectral_density_matrix(mixture_stft.transpose(1, 0, 2), mask_target, normalize=True)
    cov_noise = condition_covariance(cov_noise, 1e-6)
    cov_noise /= np.trace(cov_noise, axis1=-2, axis2=-1)[..., None, None]

    h = []
    for f in range(F):
        try:
            _cov_noise = cov_noise[f]
            _cov_speech = cov_speech[f]

            # mask-based GEV
            [_d, _v] = eigh(_cov_speech, _cov_noise)
            h.append(_v[:, -1])

        except LinAlgError:  # just a precaution if the solve does not work
            h.append(np.ones((C,)) + 1j * np.ones((C,)))

    w = np.array(h)
    if phase_correct:
        w = phase_correction(w)
    return w


def MB_GEV(mixture, noise, target, mask="IBM", frame_len=2048, frame_step=512, phase_correct=False,
                  ref_mic=0):
    """
    Mask based MVDR beamformer as formulated in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664.
    This implementation uses oracle masks but you can use mb_gev_weights to get the weights if you have custom masks
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param mask: type of oracle mask: IBM, IRM, WFM, or PSM (see https://arxiv.org/pdf/1709.00917.pdf and
    https://arxiv.org/pdf/1809.07454.pdf)
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :param ref_mic: int, (self explanatory)
    :return: the enhanced signal
    """
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    mask_target, mask_noise = calculate_masks([target_stft, noise_stft], mask=mask)

    w = mb_gev_weights(noise_stft + target_stft, mask_noise[ref_mic], mask_target[ref_mic], phase_correct=phase_correct)
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    reconstructed = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))

    return reconstructed


def MB_GEV_oracle(mixture, noise, target, mask="IBM", frame_len=2048, frame_step=512, phase_correct=False,
                  ref_mic=0):
    """
    Mask based MVDR beamformer as formulated in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664.
    This implementation uses oracle masks but you can use mb_gev_weights to get the weights if you have custom masks
    :param mixture: nd_array (n_mics, time) of the mixture recordings
    :param noise: nd_array (n_mics, time) of the noise recordings
    :param target: nd_array (n_mics, time) of the target recordings
    :param mask: type of oracle mask: IBM, IRM, WFM, or PSM (see https://arxiv.org/pdf/1709.00917.pdf and
    https://arxiv.org/pdf/1809.07454.pdf)
    :param frame_len: int (self explanatory)
    :param frame_step: int (self explanatory)
    :param phase_correct: whether or not to phase correct (see https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7471664)
    :param ref_mic: int, (self explanatory)
    :return: the enhanced signal
    """
    mixture_stft = stft(mixture, frame_len=frame_len, frame_step=frame_step)
    target_stft = stft(target, frame_len=frame_len, frame_step=frame_step)
    noise_stft = stft(noise, frame_len=frame_len, frame_step=frame_step)

    mask_target, mask_noise = calculate_masks([target_stft, noise_stft], mask=mask)

    w = mb_gev_weights(mixture_stft, mask_noise[ref_mic], mask_target[ref_mic], phase_correct=phase_correct)
    sep_spec = apply_beamforming_weights(mixture_stft, w)

    reconstructed = istft(sep_spec, frame_len=frame_len, frame_step=frame_step, input_len=len(target[ref_mic]))

    return reconstructed


def calculate_masks(signals, mask="IBM"):
    """ Calculates the ideal spectral masks for all the signals in signals
    mask can be IBM, IRM, WFM, PSM
    """
    n_sources = len(signals)
    full_set = {i for i in range(n_sources)}
    if mask == "IBM":
        masks = [np.prod([np.float32(np.abs(x) > np.abs(signals[y])) for y in list(full_set.difference({j}))], 0) for
                 j, x in enumerate(signals)]
    elif mask == "IRM":
        masks = [np.abs(x) / np.sum(np.abs(signals), 0) for x in signals]
    elif mask == "WFM":
        masks = [np.abs(x) ** 2 / np.sum(np.abs(signals) ** 2, 0) for x in signals]
    elif mask == "PSM":
        masks = [np.cos(np.angle(x) + np.angle(sum(signals))) * np.abs(x) / np.abs(sum(signals)) for x in signals]
    else:
        raise ValueError("{} not Implemented".format(mask))
    return masks


def get_power_spectral_density_matrix(observation, mask=None, normalize=True):
    """
    Calculates the weighted power spectral density matrix.
    This does not yet work with more than one target mask.
    :param normalize: wheter or not normalize the psd
    :param observation: Complex observations with shape (bins, sensors, frames)
    :param mask: Masks with shape (bins, frames) or (bins, 1, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """
    bins, sensors, frames = observation.shape

    if mask is None:
        mask = np.ones((bins, frames))
    if mask.ndim == 2:
        mask = mask[:, np.newaxis, :]

    psd = np.einsum('...dt,...et->...de', mask * observation,
                    observation.conj())
    if normalize:
        normalization = np.sum(mask, axis=-1, keepdims=True)
        psd /= normalization + 1e-15
    return psd


def phase_correction(vector):
    """Phase correction to reduce distortions due to phase inconsistencies.
    Args:
        vector: Beamforming vector with shape (..., bins, sensors).
    Returns: Phase corrected beamforming vectors. Lengths remain.
    """
    w = vector.copy()
    F, D = w.shape
    for f in range(1, F):
        w[f, :] *= np.exp(-1j * np.angle(
            np.sum(w[f, :] * w[f - 1, :].conj(), axis=-1, keepdims=True)))
    return w
