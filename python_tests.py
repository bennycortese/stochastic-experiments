import unittest
import math
from quiz_study import exponential_distribution, expectation, variance, uniform, brownian, simulate_1d_gbm
import yfinance as yf
import numpy as np

class TestStringMethods(unittest.TestCase):

    # help in game, godot

    def apple_as_geometric_brownian():
        simulate_1d_gbm(apple_nsteps=1000, apple_t=1, apple_mu=0.0001, apple_sigma=0.02, apple_start=1)
        # find some way to model apple as a geometric brownian motion
        pass

    
    def msft_as_geometric_brownian():
        # Fetch Microsoft stock data
        msft = yf.Ticker("MSFT")
        
        # Get the current closing price to use as the start value
        current_price = msft.history(period="1d")['Close'].iloc[-1]
        
        # Define parameters for the simulation
        nsteps = 1000
        t = 1  # Simulate over one year
        mu = 0.0001  # Assume a daily return rate
        sigma = 0.02  # Stock volatility

        # Time increment
        dt = t / nsteps

        # Simulate the stock price path
        price_path = [current_price]
        for _ in range(1, nsteps):
            previous_price = price_path[-1]
            shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            price = previous_price * np.exp(shock)
            price_path.append(price)

        return price_path

    def pull_recent_yahoo_data(stock_ticker):
        #cur_data = retrieval(stock_ticker)
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
