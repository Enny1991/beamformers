from __future__ import print_function, division
from unittest import TestCase
import numpy as np
import warnings
import os
import soundfile as sf

from beamformers.beamformers import MVDR, SDW_MWF, MWF_Oracle, MSNR, BeamformIt

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestBeamformers(TestCase):

    def setUp(self):

        mix, fs = sf.read('wavs/mix.wav')
        self.mix = mix.T
        nn, fs = sf.read('wavs/nn.wav')
        self.nn = nn.T
        spk, fs = sf.read('wavs/spk.wav')
        self.spk = spk.T
        gt, fs = sf.read('wavs/gt.wav')
        self.gt = gt

        self.fs = fs
        receptive_field = 0.128  # in s
        self.frame_len = int(fs * receptive_field)
        self.frame_step = int(self.frame_len / 4)

    def tearDown(self):
        # os.remove('wavs/test_out.wav')
        pass

    # def test_sdw_mwf_wref(self):
    #
    #     true_sdw_mwf_wref, fs = sf.read('wavs/sdw_mwf_wref.wav')
    #
    #     out_sdw_mwf_wref = SDW_MWF(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_sdw_mwf_wref, self.fs)
    #     out_sdw_mwf_wref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_sdw_mwf_wref, true_sdw_mwf_wref)
    #
    # def test_sdw_mwf_nref(self):
    #     true_sdw_mwf_nref, fs = sf.read('wavs/sdw_mwf_nref.wav')
    #
    #     out_sdw_mwf_nref = SDW_MWF(self.mix, self.nn, target=None,
    #                                frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_sdw_mwf_nref, self.fs)
    #     out_sdw_mwf_nref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_sdw_mwf_nref, true_sdw_mwf_nref)

    # def test_msnr_wref(self):
    #
    #     true_msnr_wref, fs = sf.read('wavs/msnr_wref.wav')
    #
    #     out_msnr_wref = MSNR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_msnr_wref, self.fs)
    #     out_msnr_wref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_msnr_wref, true_msnr_wref)
    #
    # def test_msnr_nref(self):
    #     true_msnr_nref, fs = sf.read('wavs/msnr_nref.wav')
    #
    #     out_msnr_nref = MSNR(self.mix, self.nn, target=None, frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_msnr_nref, self.fs)
    #     out_msnr_nref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_msnr_nref, true_msnr_nref)

    # def test_mvdr_wref(self):
    #
    #     true_mvdr_wref, fs = sf.read('wavs/mvdr_wref.wav')
    #
    #     out_mvdr_wref = MVDR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_mvdr_wref, self.fs)
    #     out_mvdr_wref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_mvdr_wref, true_mvdr_wref)
    #
    # def test_mvdr_nref(self):
    #     true_mvdr_nref, fs = sf.read('wavs/mvdr_nref.wav')
    #
    #     out_mvdr_nref = MVDR(self.mix, self.nn, target=None, frame_len=self.frame_len, frame_step=self.frame_step)
    #     sf.write('wavs/test_out.wav', out_mvdr_nref, self.fs)
    #     out_mvdr_nref, fs = sf.read('wavs/test_out.wav')
    #
    #     np.testing.assert_equal(out_mvdr_nref, true_mvdr_nref)

    def test_mwf(self):

        true_mwf, fs = sf.read('wavs/mwf.wav')

        out_mwf = MWF_Oracle(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_mwf, self.fs)
        out_mwf, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_mwf, true_mwf)

    # def test_bfi(self):
    #
    #     out_bfi = BeamformIt(self.mix).astype('float64')
    #
    #     sf.write('wavs/tout.wav', out_bfi, 8000)
    #
    #     true_bfi, fs = sf.read('wavs/bfi.wav')
    #     out_bfi, fs = sf.read('wavs/tout.wav')
    #
    #     np.testing.assert_equal(out_bfi, true_bfi)


