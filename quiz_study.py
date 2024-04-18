import numpy as np
import math
from math import sqrt
from scipy.stats import norm


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
    return math.exp(-t*λ) * (λ * t) ** k / factorial(k)

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
