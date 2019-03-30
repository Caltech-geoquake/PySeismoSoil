# Author: Jian Shi

import unittest
import numpy as np

from PySeismoSoil.class_site_effect_adjustment import Site_Effect_Adjustment
from PySeismoSoil.class_ground_motion import Ground_Motion

class Test_Class_Site_Effect_Adjustment(unittest.TestCase):

    def test_init(self):
        gm = Ground_Motion('./files/sample_accel.txt', unit='gal')
        vs30 = 250
        z1 = 150
        Site_Effect_Adjustment(gm, vs30, z1)

    def test_run(self):
        gm_in = Ground_Motion('./files/sample_accel.txt', unit='gal')
        vs30 = 207
        z1 = 892
        sea = Site_Effect_Adjustment(gm_in, vs30, z1)
        gm_out = sea.run(show_fig=True, dpi=150)
        self.assertTrue(isinstance(gm_out, Ground_Motion))

        # Test out-of-bound Vs30 (lenient cases)
        sea1 = Site_Effect_Adjustment(gm_in, 170, 75, lenient=True)
        sea2 = Site_Effect_Adjustment(gm_in, 175, 75)
        motion_out1 = sea1.run()
        motion_out2 = sea2.run()
        self.assertTrue(np.allclose(motion_out1.accel, motion_out2.accel))

        # Test out-of-bound z1 (lenient cases)
        sea1 = Site_Effect_Adjustment(gm_in, 360, 927, lenient=True)
        sea2 = Site_Effect_Adjustment(gm_in, 360, 900)
        motion_out1 = sea1.run()
        motion_out2 = sea2.run()
        self.assertTrue(np.allclose(motion_out1.accel, motion_out2.accel))

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Site_Effect_Adjustment)
    unittest.TextTestRunner(verbosity=2).run(SUITE)