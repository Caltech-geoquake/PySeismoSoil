# Author: Jian Shi

import unittest

from PySeismoSoil.class_simulation import Linear_Simulation
from PySeismoSoil.class_ground_motion import Ground_Motion
from PySeismoSoil.class_Vs_profile import Vs_Profile

class Test_Class_Simulation(unittest.TestCase):
    def test_linear(self):
        input_motion = Ground_Motion('./files/sample_accel.txt', unit='m')
        soil_profile = Vs_Profile('./files/profile_FKSH14.txt')
        ls = Linear_Simulation(input_motion, soil_profile)
        output = ls.run(show_fig=True)

        self.assertEqual(output.accel.shape, input_motion.accel.shape)
        self.assertEqual(output.dt, input_motion.dt)
        self.assertEqual(output.npts, input_motion.npts)

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_Simulation)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
