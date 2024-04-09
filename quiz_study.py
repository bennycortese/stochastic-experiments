import numpy as np

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