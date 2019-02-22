# Author: Jian Shi

import unittest

from PySeismoSoil.class_simulator import Linear_Simulator
from PySeismoSoil.class_ground_motion import Ground_Motion

class Test_Class_Simulator(unittest.TestCase):
    def test_linear(self):
        input_motion = './files/sample_accel.txt'
        soil_profile = './files/profile_FKSH14.txt'
        ls = Linear_Simulator(input_motion, soil_profile)
        output = ls.run()

        gm_input = Ground_Motion(input_motion, 'm/s/s')
        self.assertEqual(output.accel.shape, gm_input.accel.shape)
        self.assertEqual(output.dt, gm_input.dt)
        self.assertEqual(output.npts, gm_input.npts)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Simulator)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
