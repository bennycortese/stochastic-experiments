import numpy as np
import math
from math import sqrt
from scipy.stats import norm
import scipy.stats as stats
import yfinance as yf
import matplotlib.pyplot as plt

def simulate_brownian():
    pass
    # function here for n iterations and matplot

def new_distribution():
    pass

def generate_many_paths_visualization(paths):
    #generate the visual for many simulated paths from a given stock
    pass

def generate_poisson_process(rate, max_time):
    arrival_times = []
    current_time = 0
    
    # Generate events until the maximum observation time is reached
    while current_time <= max_time:
        interarrival_time = np.random.exponential(1/rate)
        current_time += interarrival_time
        if current_time <= max_time:
            arrival_times.append(current_time)
    
    return arrival_times

def calculate_average_interarrival_time(arrival_times):
    if len(arrival_times) < 2:
        return None  # Not enough data to calculate a meaningful average interarrival time
    interarrival_times = np.diff([0] + arrival_times)
    average_interarrival_time = np.mean(interarrival_times)
    return average_interarrival_time

# Example usage:
rate = 1/15  # Rate of the Poisson process
max_time = 10000  # Extended observation period for more data
arrival_times = generate_poisson_process(rate, max_time)

average_interarrival_time = calculate_average_interarrival_time(arrival_times)
if average_interarrival_time:
    print("Average interarrival time:", average_interarrival_time)
else:
    print("Insufficient data to calculate an average interarrival time.")


def exponential_distribution(λ, x):
    return λ * math.exp(-λ * x)

def cdf(λ, x):
    if x >= 0:
        return 1 - math.exp(λ * x)
    else:
        return 0
    
def expectation(λ):
    return 1 / λ

def variance(λ):
    return 1 / λ**2

def integer_part(λ, n):
    return math.exp(- λ * (n + 1))

def uniform(x): # for P(e^λ*x <= x)
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        return x

#def simulate(n):
    #for n in simulate, random process

def pdf_with_t(t, λ, k):
    return math.exp(-t*λ) * (λ * t) ** k / math.factorial(k)

def brownian_motion_with_t(t, λ):
    return np.random.exponential(λ * t)

def brownian(x0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.
    
    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

print(exponential_distribution(1, 1))

print(cdf(-1, 1))

def brownian_motion_probability():
    t = 1  # Example time
    mean = 0
    variance = t
    std_dev = variance**0.5

    # Calculate the probability that B(t) >= 0
    probability = 1 - stats.norm.cdf(0, loc=mean, scale=std_dev)
    print("The probability that B(t) >= 0 is:", probability)
        

def simulate_1d_gbm(nsteps=1000, t=1, mu=0.0001, sigma=0.02, start=1):
    """
    Simulates the 1D geometric Brownian motion process.

    Parameters:
        nsteps (int): The number of steps in the simulation. Default is 1000.
        t (float): The time horizon for the simulation. Default is 1.
        mu (float): The drift parameter of the geometric Brownian motion process. Default is 0.0001.
        sigma (float): The volatility parameter of the geometric Brownian motion process. Default is 0.02.
        start (float): The initial value of the process. Default is 1.

    Returns:
        x (list): A list of time points corresponding to the simulation steps.
        y (ndarray): An array of values representing the simulated geometric Brownian motion process.
    """
    steps = [ (mu - (sigma**2)/2) + np.random.randn()*sigma for i in range(nsteps) ]
    y = start*np.exp(np.cumsum(steps))
    x = [ t*i for i in range(nsteps) ]
    return x, y

def visualize_prediction(data_x, data_y):
        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.plot(data_x, data_y, marker='o', linestyle='-', color='b')  # Plot data points
        plt.title('Visualization of Predictions')  # Set title
        plt.xlabel('X-axis')  # Label x-axis
        plt.ylabel('Y-axis')  # Label y-axis
        plt.grid(True)  # Enable grid
        plt.show()  # Display the plot
    
    # help in game, godot

def apple_as_geometric_brownian(stock_ticker):
        # Fetch Apple stock data
        apple = yf.Ticker("AAPL") # make this a parameter 
        
        # Get the current closing price to use as the start value
        current_price = apple.history(period="1d")['Close'].iloc[-1]
        
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
apple_prices = []
for i in range(1000):
    apple_prices.append(apple_as_geometric_brownian())

<<<<<<< HEAD
visualize_prediction(range(len(apple_prices)), apple_prices)
=======
apple_prices = []
for i in range(1000):
    apple_prices.append(apple_as_geometric_brownian())
visualize_prediction(range(len(apple_prices)), apple_prices)
>>>>>>> 77efdcc783ea11d4514a76471070d319c606cc4b
