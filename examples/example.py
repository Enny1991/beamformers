from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
import soundfile as sf

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from beamformers import beamformers

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():

    # parameters
    fs = 8000
    receptive_field = 0.128  # in s
    frame_len = int(fs * receptive_field)
    frame_step = int(frame_len / 4)

    # TARGET / REFERENCE
    # First a load a multi-channel wav file, of course you can use any array of shape (n_mics, time)
    spk, _ = sf.read('../wavs/spk.wav')
    # soundfile loads the file is as (time, n_mics) so I need to transpose it
    spk = spk.T

    # INTERFERENCE
    nn, _ = sf.read('../wavs/nn.wav')
    # soundfile loads the file is as (time, n_mics) so I need to transpose it
    nn = nn.T

    # GROUNDTRUTH
    gt, _ = sf.read('../wavs/gt.wav')

    # MIXTURE: REFERENCE + INTERFERENCE
    mix, fs = sf.read('../wavs/mix.wav')
    # soundfile loads the file is as (time, n_mics) so I need to transpose it
    mix = mix.T

    # SDW_MWF - A
    print("SDW_MWF - With reference")
    out_sdwmwf_a = beamformers.SDW_MWF(mix, nn, spk, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_sdwmwf_a.wav', out_sdwmwf_a, fs)

    # SDW_MWF - B
    print("SDW_MWF - Without reference")
    out_sdwmwf_b = beamformers.SDW_MWF(mix, nn, reference=None, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_sdwmwf_b.wav', out_sdwmwf_b, fs)

    # MWF
    print("Oracle MWF")
    out_mwf = beamformers.MWF_Oracle(mix, spk, nn, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_mwf.wav', out_mwf, fs)

    # MVDR - A
    print("MVDR - With reference")
    out_mvdr_a = beamformers.MVDR(mix, nn, spk, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_mvdr_a.wav', out_mvdr_a, fs)

    # MVDR - B
    print("MVDR - Without reference")
    out_mvdr_b = beamformers.MVDR(mix, nn, reference=None, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_mvdr_b.wav', out_mvdr_b, fs)

    # MSNR - A
    print("MSNR - Without reference")
    out_msnr_a = beamformers.MSNR(mix, nn, spk, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_msnr_a.wav', out_msnr_a, fs)

    # MSNR - B
    print("MSNR - Without reference")
    out_msnr_b = beamformers.MSNR(mix, nn, reference=None, frame_len=frame_len, frame_step=frame_step)
    sf.write('../wavs/out_msnr_b.wav', out_msnr_b, fs)

    # TD-MVDR - A
    print("TD_MVDR - With reference (takes a long time)")
    out_tdmvdr_a = beamformers.TD_MVDR(mix, nn, spk, frame_len=frame_len)
    sf.write('../wavs/out_tdmvdr_a.wav', out_tdmvdr_a, fs)

    # TD-MVDR - B
    print("TD_MVDR - Without reference (takes a long time)")
    out_tdmvdr_b = beamformers.TD_MVDR(mix, nn, reference=None, frame_len=frame_len)
    sf.write('../wavs/out_tdmvdr_b.wav', out_tdmvdr_b, fs)

    # TD-MWF - A
    print("TD_MWF - With reference (takes a long time)")
    out_tdmwf_a = beamformers.TD_MWF(mix, nn, spk, frame_len=frame_len)
    sf.write('../wavs/out_tdmwf_a.wav', out_tdmwf_a, fs)

    # TD-MWF - B
    print("TD_MWF - Without reference (takes a long time)")
    out_tdmwf_b = beamformers.TD_MWF(mix, nn, reference=None, frame_len=frame_len)
    sf.write('../wavs/out_tdmwf_b.wav', out_tdmwf_b, fs)

    # BeamformIt
    print("BeamformIt")
    out_bfi = beamformers.BeamformIt(mix)
    sf.write('../wavs/out_bfi.wav', out_bfi, fs)


if __name__ == '__main__':
    main()
