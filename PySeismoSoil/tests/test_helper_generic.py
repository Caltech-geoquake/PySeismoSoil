# Author: Jian Shi

import unittest
import numpy as np

import PySeismoSoil.helper_generic as hlp

class Test_Helper_Generic(unittest.TestCase):
    '''
    Unit test for helper functions in helper_generic.py
    '''

    def test_read_two_column_stuff(self):

        # Load from file
        data, dt = hlp._read_two_column_stuff('two_column_data_example.txt')
        benchmark = np.array([[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0,
                               1.1, 1.2, 1.3, 1.4, 1.5],
                              [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7]]).T
        self.assertTrue(np.allclose(data, benchmark))
        self.assertAlmostEqual(dt, benchmark[1, 0] - benchmark[0, 0])

        # Load from 1D numpy array
        data_1col = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7])
        dt = 0.1
        data_, dt_ = hlp._read_two_column_stuff(data, dt)
        self.assertTrue(np.allclose(data_1col, data_[:, 1]))
        self.assertAlmostEqual(dt, dt_)

        # Load from 2D numpy array
        data__, dt__ \
            = hlp._read_two_column_stuff(benchmark,
                                         benchmark[1, 0] - benchmark[0, 0])
        self.assertTrue(np.allclose(data__, benchmark))
        self.assertAlmostEqual(dt__, benchmark[1, 0] - benchmark[0, 0])

    def test_check_two_column_format(self):

        with self.assertRaisesRegex(TypeError, '1.5 should be an numpy array.'):
            hlp._check_two_column_format(1.5, '1.5')
        with self.assertRaisesRegex(TypeError, '_a_ should be a 2D numpy array.'):
            hlp._check_two_column_format(np.array([1,2,3]), '_a_')
        with self.assertRaisesRegex(TypeError, '_b_ should have two columns.'):
            hlp._check_two_column_format(np.ones((2, 3)), '_b_')

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Helper_Generic)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
