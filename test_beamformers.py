from __future__ import print_function, division
from unittest import TestCase
import pkg_resources
import logging
import numpy as np
import warnings
import soundfile as sf

from beamformers import MVDR, SDW_MWF, MWF_Oracle, MSNR, TD_MVDR, TD_MWF

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestDeepExplainGeneralTF(TestCase):

    def setUp(self):

        mix, fs = sf.read('wavs/mix.wav')
        self.mix = mix.T
        nn, fs = sf.read('wavs/nn.wav')
        self.nn = nn.T
        spk, fs = sf.read('wavs/spk.wav')
        self.spk = spk.T
        gt, fs = sf.read('wavs/gt.wav')
        self.gt = gt

        fs = 8000
        receptive_field = 0.128  # in s
        self.frame_len = int(fs * receptive_field)
        self.frame_step = int(self.frame_len / 4)

    # def tearDown(self):
    #     print("done")
    
    def test_td_mwf_wref(self):

        true_td_mwf_wref, fs = sf.read('wavs/td_mwf_wref.wav')

        out_td_mwf_wref = TD_MWF(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_td_mwf_wref, 8000)
        out_td_mwf_wref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_td_mwf_wref, true_td_mwf_wref)

    def test_td_mwf_nref(self):
        true_td_mwf_nref, fs = sf.read('wavs/td_mwf_nref.wav')

        out_td_mwf_nref = TD_MWF(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_td_mwf_nref, 8000)
        out_td_mwf_nref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_td_mwf_nref, true_td_mwf_nref)
    
    def test_td_mvdr_wref(self):

        true_td_mvdr_wref, fs = sf.read('wavs/td_mvdr_wref.wav')

        out_td_mvdr_wref = TD_MVDR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_td_mvdr_wref, 8000)
        out_td_mvdr_wref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_td_mvdr_wref, true_td_mvdr_wref)

    def test_td_mvdr_nref(self):
        true_td_mvdr_nref, fs = sf.read('wavs/td_mvdr_nref.wav')

        out_td_mvdr_nref = TD_MVDR(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_td_mvdr_nref, 8000)
        out_td_mvdr_nref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_td_mvdr_nref, true_td_mvdr_nref)
        
    def test_sdw_mwf_wref(self):

        true_sdw_mwf_wref, fs = sf.read('wavs/sdw_mwf_wref.wav')

        out_sdw_mwf_wref = SDW_MWF(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_sdw_mwf_wref, 8000)
        out_sdw_mwf_wref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_sdw_mwf_wref, true_sdw_mwf_wref)

    def test_sdw_mwf_nref(self):
        true_sdw_mwf_nref, fs = sf.read('wavs/sdw_mwf_nref.wav')

        out_sdw_mwf_nref = SDW_MWF(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_sdw_mwf_nref, 8000)
        out_sdw_mwf_nref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_sdw_mwf_nref, true_sdw_mwf_nref)
        
    def test_msnr_wref(self):

        true_msnr_wref, fs = sf.read('wavs/msnr_wref.wav')

        out_msnr_wref = MSNR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_msnr_wref, 8000)
        out_msnr_wref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_msnr_wref, true_msnr_wref)

    def test_msnr_nref(self):
        true_msnr_nref, fs = sf.read('wavs/msnr_nref.wav')

        out_msnr_nref = MSNR(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_msnr_nref, 8000)
        out_msnr_nref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_msnr_nref, true_msnr_nref)

    def test_mvdr_wref(self):

        true_mvdr_wref, fs = sf.read('wavs/mvdr_wref.wav')

        out_mvdr_wref = MVDR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_mvdr_wref, 8000)
        out_mvdr_wref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_mvdr_wref, true_mvdr_wref)

    def test_mvdr_nref(self):
        true_mvdr_nref, fs = sf.read('wavs/mvdr_nref.wav')

        out_mvdr_nref = MVDR(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_mvdr_nref, 8000)
        out_mvdr_nref, fs = sf.read('wavs/test_out.wav')

        np.testing.assert_equal(out_mvdr_nref, true_mvdr_nref)

    def test_sdw_mwf(self):

        true_mwf, fs = sf.read('wavs/mwf.wav')

        out_mwf = MWF_Oracle(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        sf.write('wavs/test_out.wav', out_mwf, 8000)
        out_mwf, fs = sf.read('wavs/test_out.wav')
        
        np.testing.assert_equal(out_mwf, true_mwf)

