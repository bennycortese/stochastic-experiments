import unittest
from quiz_study import exponential_distribution, expectation, variance, uniform

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def brownian_test(self)
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

    def test_expectation(self):
        self.assertEqual(exponential_distribution(1, 1), 0.36787944117144233)

    def test_exponential_expectation(self):
        self.assertEqual(expectation(2), 1/2)

    def test_exponential_variation(self):
        self.assertEqual(variance(2), 1/4)

    def test_uniform(self):
        self.assertEqual(uniform(2), 1)

    def test_uniform(self):
        self.assertEqual(uniform(3), 1)

    def test_exponential_variation(self):
        self.assertEqual(variance(3), 1/8)

if __name__ == '__main__':
    unittest.main()
