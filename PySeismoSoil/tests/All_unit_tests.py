# Author: Jian Shi

import unittest

from test_class_batch_simulation import Test_Class_Batch_Simulation
from test_class_curves import Test_Class_Curves
from test_class_damping_calibration import Test_Class_Damping_Calibration
from test_class_frequency_spectrum import Test_Class_Frequency_Spectrum
from test_class_ground_motion import Test_Class_Ground_Motion
from test_class_hh_calibration import Test_Class_HH_Calibration
from test_class_parameters import Test_Class_HH_Param
from test_class_simulation import Test_Class_Simulation
from test_class_simulation_results import Test_Class_Simulation_Results
from test_class_site_effect_adjustment import Test_Class_Site_Effect_Adjustment
from test_class_site_factors import Test_Class_Site_Factors
from test_class_svm import Test_Class_SVM
from test_class_Vs_profile import Test_Class_Vs_Profile

from test_helper_generic import Test_Helper_Generic
from test_helper_hh_calibration import Test_Helper_HH_Calibration
from test_helper_hh_model import Test_Helper_HH_Model
from test_helper_mkz_model import Test_Helper_MKZ_Model
from test_helper_signal_processing import Test_Helper_Signal_Processing
from test_helper_simulations import Test_Helper_Simulations
from test_helper_site_response import Test_Helper_Site_Response

TS = unittest.TestSuite()

TS.addTests(unittest.makeSuite(Test_Class_Batch_Simulation))
TS.addTests(unittest.makeSuite(Test_Class_Curves))
TS.addTests(unittest.makeSuite(Test_Class_Damping_Calibration))
TS.addTests(unittest.makeSuite(Test_Class_Frequency_Spectrum))
TS.addTests(unittest.makeSuite(Test_Class_Ground_Motion))
TS.addTests(unittest.makeSuite(Test_Class_HH_Calibration))
TS.addTests(unittest.makeSuite(Test_Class_HH_Param))
TS.addTests(unittest.makeSuite(Test_Class_Simulation))
TS.addTests(unittest.makeSuite(Test_Class_Simulation_Results))
TS.addTests(unittest.makeSuite(Test_Class_Site_Effect_Adjustment))
TS.addTests(unittest.makeSuite(Test_Class_Site_Factors))
TS.addTests(unittest.makeSuite(Test_Class_SVM))
TS.addTests(unittest.makeSuite(Test_Class_Vs_Profile))

TS.addTests(unittest.makeSuite(Test_Helper_Generic))
TS.addTests(unittest.makeSuite(Test_Helper_HH_Calibration))
TS.addTests(unittest.makeSuite(Test_Helper_HH_Model))
TS.addTests(unittest.makeSuite(Test_Helper_MKZ_Model))
TS.addTests(unittest.makeSuite(Test_Helper_Signal_Processing))
TS.addTests(unittest.makeSuite(Test_Helper_Simulations))
TS.addTests(unittest.makeSuite(Test_Helper_Site_Response))

if __name__ == '__main__':
    print('------ Running all unit tests. -------')
    unittest.TextTestRunner(verbosity=2).run(TS)
