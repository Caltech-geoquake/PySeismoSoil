import unittest

import numpy as np

from PySeismoSoil.class_svm import SVM
from PySeismoSoil.class_Vs_profile import Vs_Profile


class Test_Class_SVM(unittest.TestCase):
    def test_init(self):
        Vs30 = 256
        z1 = 100
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        self.assertEqual(svm.Vs30, Vs30)
        self.assertEqual(svm.z1, z1)

    def test_Vs_cap_is_True(self):
        Vs30 = 256
        z1 = 10
        svm = SVM(Vs30, z1=z1, Vs_cap=True)
        self.assertEqual(0, svm.base_profile.vs_profile[-1, 0])
        self.assertEqual(1000, svm.base_profile.vs_profile[-1, 1])

    def test_Vs_cap_is_user_defined(self):
        Vs30 = 256
        z1 = 10
        svm = SVM(Vs30, z1=z1, Vs_cap=1234.5)
        self.assertEqual(0, svm.base_profile.vs_profile[-1, 0])
        self.assertEqual(1234.5, svm.base_profile.vs_profile[-1, 1])

    def test_Vs_cap_is_False(self):
        pass  # this case is hard to test; skipped for now

    def test_base_profile(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        base_profile = svm.base_profile
        self.assertTrue(isinstance(base_profile, Vs_Profile))

    def test_get_discretized_profile__fixed_thk(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        discr_profile = svm.get_discretized_profile(
            fixed_thk=10, show_fig=False
        )
        self.assertTrue(isinstance(discr_profile, Vs_Profile))
        if svm.has_bedrock_Vs:  # bedrock Vs must match
            self.assertEqual(svm.bedrock_Vs, discr_profile.vs_profile[-1, 1])
            self.assertEqual(0, discr_profile.vs_profile[-1, 0])

    def test_get_discretized_profile__valid_Vs_increment(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        discr_profile = svm.get_discretized_profile(
            Vs_increment=100, show_fig=False
        )
        self.assertTrue(isinstance(discr_profile, Vs_Profile))
        if svm.has_bedrock_Vs:  # bedrock Vs must match
            self.assertEqual(svm.bedrock_Vs, discr_profile.vs_profile[-1, 1])
            self.assertEqual(0, discr_profile.vs_profile[-1, 0])

    def test_get_discretized_profile__invalid_Vs_increment(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with self.assertRaisesRegex(
            ValueError, 'max Vs of the smooth profile'
        ):
            svm.get_discretized_profile(Vs_increment=5000)

    def test_get_discretized_profile__both_input_param_are_None(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with self.assertRaisesRegex(ValueError, 'You need to provide either'):
            svm.get_discretized_profile(Vs_increment=None, fixed_thk=None)

    def test_get_discretized_profile__both_input_param_are_provided(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        with self.assertRaisesRegex(ValueError, 'do not provide both'):
            svm.get_discretized_profile(Vs_increment=1, fixed_thk=2)

    def test_get_randomized_profile(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)
        random_profile = svm.get_randomized_profile(show_fig=False)
        self.assertTrue(isinstance(random_profile, Vs_Profile))

        if svm.has_bedrock_Vs:  # bedrock Vs must match
            self.assertEqual(svm.bedrock_Vs, random_profile.vs_profile[-1, 1])
            self.assertEqual(0, random_profile.vs_profile[-1, 0])

        # Use iteration to pick compliant randomized Vs profile
        random_profile = svm.get_randomized_profile(
            show_fig=True,
            vs30_z1_compliance=True,
            verbose=True,
        )

    def test_get_randomized_profile__ends_with_bedrock(self):
        # The randomized Vs of the last soil layer can be higher than the
        # bedrock Vs. The profile used to end without the bedrock in this case,
        # which made `test_get_randomized_profile()` fail randomly.
        svm = SVM(target_Vs30=256, z1=200, show_fig=False)
        n_last_layer_stiffer_than_bedrock = 0
        for seed in range(30):
            profile = svm.get_randomized_profile(seed=seed).vs_profile
            np.testing.assert_array_equal([0, svm.bedrock_Vs], profile[-1, :2])
            self.assertAlmostEqual(svm.z1, np.sum(profile[:, 0]))
            self.assertTrue(np.all(profile[:-1, 0] >= 2.0))  # no thin layers
            if profile[-2, 1] >= svm.bedrock_Vs:
                n_last_layer_stiffer_than_bedrock += 1

        # Make sure that the seeds above do cover this case
        self.assertGreater(n_last_layer_stiffer_than_bedrock, 0)

    def test_get_randomized_profile__Vs_cap_is_False(self):
        # Without bedrock, the randomized profile ends with the same half-space
        # as the base profile (it used to always end with 1000 m/s, so the
        # loop for a compliant profile could never end)
        svm = SVM(target_Vs30=256, z1=100, Vs_cap=False, show_fig=False)
        base_halfspace = svm.base_profile.vs_profile[-1, :2]
        self.assertLess(base_halfspace[1], 1000)
        for seed in range(5):
            profile = svm.get_randomized_profile(seed=seed).vs_profile
            np.testing.assert_array_equal(base_halfspace, profile[-1, :2])

    def test_get_randomized_profile__seed(self):
        svm = SVM(target_Vs30=256, z1=100, show_fig=False)

        # The same seed gives the same profile
        profile_1 = svm.get_randomized_profile(seed=5).vs_profile
        profile_2 = svm.get_randomized_profile(seed=5).vs_profile
        np.testing.assert_array_equal(profile_1, profile_2)

        # Without a seed, every call gives a different profile. (The seed used
        # to come from the current time in seconds, so all the calls within
        # the same second gave the same profile.)
        profile_3 = svm.get_randomized_profile().vs_profile
        profile_4 = svm.get_randomized_profile().vs_profile
        self.assertFalse(np.array_equal(profile_3, profile_4))

        # numpy's global random state is not changed
        np.random.seed(0)
        expected = np.random.random()
        np.random.seed(0)
        svm.get_randomized_profile(seed=5)
        svm.get_randomized_profile()
        self.assertEqual(expected, np.random.random())

    def test_index_closest(self):
        array = [0, 1, 2, 1.1, 0.4, -3.2]
        i, val = SVM._find_index_closest(array, 2.1)
        self.assertEqual((2, 2), (i, val))

        i, val = SVM._find_index_closest(array, -9)
        self.assertEqual((5, -3.2), (i, val))

        i, val = SVM._find_index_closest(array, -0.5)
        self.assertEqual((0, 0), (i, val))

        i, val = SVM._find_index_closest([1], 10000)
        self.assertEqual((0, 1), (i, val))

        with self.assertRaisesRegex(ValueError, 'length of `array` needs to'):
            SVM._find_index_closest([], 2)

    def test_svm_profiles_plotting(self):
        vs30 = 256  # m/s
        z1 = 200  # m
        svm = SVM(vs30, z1=z1, show_fig=False)
        svm.get_discretized_profile(fixed_thk=20, show_fig=True)
        svm.get_discretized_profile(Vs_increment=1, show_fig=True)
        svm.get_discretized_profile(Vs_increment=100, show_fig=True)
        svm.get_randomized_profile(show_fig=True)


if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Class_SVM)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
