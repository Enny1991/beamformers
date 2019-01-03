from __future__ import print_function, division
from unittest import TestCase
import pkg_resources
import logging
import numpy as np
import warnings
import soundfile as sf

from beamformers import MVDR, SDW_MWF

warnings.simplefilter(action='ignore', category=FutureWarning)


class TestDeepExplainGeneralTF(TestCase):

    def setUp(self):
        print("starting")
        mix, fs = sf.read('wavs/mix.wav')
        self.mix = mix.T
        nn, fs = sf.read('wavs/nn.wav')
        self.nn = nn.T
        spk, fs = sf.read('wavs/spk.wav')
        self.spk = spk.T
        gt, fs = sf.read('wavs/gt.wav')
        self.gt = gt

        fs = 8000
        receptive_field = 0.064  # in s
        self.frame_len = int(fs * receptive_field)
        self.frame_step = int(self.frame_len / 4)

    def tearDown(self):
        print("done")

    def test_mvdr(self):

        true_mvdr_wref, fs = sf.read('wavs/mvdr_wref.wav')
        true_mvdr_nref, fs = sf.read('wavs/mvdr_nref.wav')

        out_mvdr_wref = MVDR(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        out_mvdr_wref /= np.max(np.abs(out_mvdr_wref))
        out_mvdr_nref = MVDR(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)

        # self.assertEqual(out_mvdr_wref, true_mvdr_wref)
        # self.assertEqual(out_mvdr_nref, true_mvdr_nref)
        np.testing.assert_equal(out_mvdr_wref, true_mvdr_wref)
        np.testing.assert_equal(out_mvdr_nref, true_mvdr_nref)

    def test_sdw_mwf(self):

        true_sdw_mwf_wref = sf.read('wavs/sdw_wref.wav')
        true_sdw_mwf_nref = sf.read('wavs/sdw_nref.wav')

        out_sdw_mwf_wref = SDW_MWF(self.mix, self.nn, self.spk, frame_len=self.frame_len, frame_step=self.frame_step)
        out_sdw_mwf_nref = SDW_MWF(self.mix, self.nn, reference=None, frame_len=self.frame_len, frame_step=self.frame_step)

        self.assertEqual(out_sdw_mwf_wref, true_sdw_mwf_wref)
        self.assertEqual(out_sdw_mwf_nref, true_sdw_mwf_nref)
        np.testing.assert_equal(out_sdw_mwf_wref, true_sdw_mwf_wref)
        np.testing.assert_equal(out_sdw_mwf_nref, true_sdw_mwf_nref)

