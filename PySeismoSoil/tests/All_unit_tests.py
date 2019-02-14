# Author: Jian Shi

import unittest

from test_class_Vs_profile import Test_Class_Vs_Profile
from test_class_ground_motion import Test_Class_Ground_Motion
from test_class_frequency_spectrum import Test_Class_Frequency_Spectrum

from test_helper_hh_model import Test_Helper_HH_Model
from test_helper_generic import Test_Helper_Generic
from test_helper_site_response import Test_Helper_Site_Response
from test_helper_signal_processing import Test_Helper_Signal_Processing

TS = unittest.TestSuite()

TS.addTests(unittest.makeSuite(Test_Class_Vs_Profile))
TS.addTests(unittest.makeSuite(Test_Class_Frequency_Spectrum))
TS.addTests(unittest.makeSuite(Test_Class_Ground_Motion))
TS.addTests(unittest.makeSuite(Test_Helper_Generic))
TS.addTests(unittest.makeSuite(Test_Helper_HH_Model))
TS.addTests(unittest.makeSuite(Test_Helper_Site_Response))
TS.addTests(unittest.makeSuite(Test_Helper_Signal_Processing))

print('------ Running all unit tests. -------')
unittest.TextTestRunner(verbosity=2).run(TS)
