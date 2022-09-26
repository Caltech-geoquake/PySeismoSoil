import unittest
import numpy as np

from PySeismoSoil.class_site_effect_adjustment import SiteEffectAdjustment
from PySeismoSoil.class_ground_motion import GroundMotion

import os
from os.path import join as _join


f_dir = _join(os.path.dirname(os.path.realpath(__file__)), 'files')


class Test_Class_Site_Effect_Adjustment(unittest.TestCase):
    def test_init(self):
        gm = GroundMotion(_join(f_dir, 'sample_accel.txt'), unit='gal')
        vs30 = 250
        z1 = 150
        SiteEffectAdjustment(gm, vs30, z1)

    def test_run__normal_case(self):
        gm_in = GroundMotion(_join(f_dir, 'sample_accel.txt'), unit='gal')
        vs30 = 207
        z1 = 892
        sea = SiteEffectAdjustment(gm_in, vs30, z1)
        gm_out = sea.run(show_fig=True, dpi=150)
        self.assertTrue(isinstance(gm_out, GroundMotion))

    def test_run__out_of_bound_Vs30_lenient_case(self):
        gm_in = GroundMotion(_join(f_dir, 'sample_accel.txt'), unit='gal')
        sea1 = SiteEffectAdjustment(gm_in, 170, 75, lenient=True)
        sea2 = SiteEffectAdjustment(gm_in, 175, 75)
        motion_out1 = sea1.run()
        motion_out2 = sea2.run()
        self.assertTrue(np.allclose(motion_out1.accel, motion_out2.accel))

    def test_run__out_of_bound_z1_lenient_case(self):
        gm_in = GroundMotion(_join(f_dir, 'sample_accel.txt'), unit='gal')
        sea1 = SiteEffectAdjustment(gm_in, 360, 927, lenient=True)
        sea2 = SiteEffectAdjustment(gm_in, 360, 900)
        motion_out1 = sea1.run()
        motion_out2 = sea2.run()
        self.assertTrue(np.allclose(motion_out1.accel, motion_out2.accel))


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Site_Effect_Adjustment)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
