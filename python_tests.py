import unittest
import math
from quiz_study import exponential_distribution, expectation, variance, uniform, brownian, simulate_1d_gbm

class TestStringMethods(unittest.TestCase):


    def apple_as_geometric_brownian():
        simulate_1d_gbm(apple_nsteps=1000, apple_t=1, apple_mu=0.0001, apple_sigma=0.02, apple_start=1)
        # find some way to model apple as a geometric brownian motion
        pass

    def msft_as_geometric_brownian():
        simulate_1d_gbm(msft_nsteps=1000, msft_t=1, msft_mu=0.0001, msft_sigma=0.02, msft_start=1)
        # find some way to model apple as a geometric brownian motion
        pass

    def pull_recent_yahoo_data():
        pass

    def model_some_poisson_process():
        # find a model like average phone time rings or something
        pass
    
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def brownian_test(self):
        self.assertEqual(brownian(1, 1, 1, 1), 0.36787944117144233)

    def brownian_expecation_test(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_expectation_distribution_1(self):
        self.assertEqual(exponential_distribution(1, 1), 0.36787944117144233)
    

    def test_expectation_e(self):
        self.assertEqual(exponential_distribution(math.exp(1), 1), 0.1793740787340172)

    def test_expectation_distribution_2(self):
        self.assertEqual(exponential_distribution(1, 4), 0.01831563888873418)


    def test_exponential_expectation(self):
        self.assertEqual(expectation(2), 1/2)

    def test_exponential_variation(self):
        self.assertEqual(variance(2), 1/4)

    def test_uniform(self):
        self.assertEqual(uniform(2), 1)

    def test_uniform(self):
        self.assertEqual(uniform(3), 1)

    def test_exponential_variation(self):
        self.assertEqual(variance(3), 1/9)

if __name__ == '__main__':
    unittest.main()
