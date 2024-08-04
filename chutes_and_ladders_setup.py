import numpy as np
import matplotlib.pyplot as plt

# Define the size of the board and create an empty transition matrix
board_size = 100
P = np.zeros((board_size, board_size))

# taking a chill day, also I want to simulate a smaller board
# More monte carlo!

L = np.zeros((board_size//10, board_size//10))

# Define a simplified version of chutes (snakes) and ladders
chutes_and_ladders = {
    1: 38, 4: 14, 9: 31, 16: 6, 21: 42, 36: 44,
    47: 26, 49: 11, 56: 53, 62: 19, 64: 60,
    71: 91, 80: 100, 87: 24, 93: 73, 95: 75,
    98: 78,
}

# Populate the transition matrix
for i in range(board_size):
    for roll in range(1, 7):  # Possible die rolls
        if i + roll < board_size:  # Adjusted to prevent out-of-bounds
            # Check if the square leads to a chute or ladder
            if i + roll in chutes_and_ladders:
                # Move directly to the chute or ladder's destination
                P[i, chutes_and_ladders[i + roll] - 1] += 1/6
            else:
                # Normal move
                P[i, i + roll] += 1/6
        else:
            # Redistribute the probabilities of rolls that exceed the board size
            P[i, board_size - 1] += 1/6

# The last square is an absorbing state
P[board_size - 1, board_size - 1] = 1

def probability_after_n_rolls(P, n):
    # Raise the transition matrix to the power of n
    P_n = np.linalg.matrix_power(P, n)
    # Return the first row, which corresponds to probabilities starting from square 0
    return P_n[0, :]

n = 3  # Number of rolls
probabilities = probability_after_n_rolls(P, n)

# Visualization
squares = np.arange(1, board_size + 1)
plt.figure(figsize=(20, 5))
plt.bar(squares, probabilities, color='skyblue')
plt.title('Probability of Landing on Each Square After 3 Rolls')
plt.xlabel('Board Squares')
plt.ylabel('Probability')
plt.xticks(np.arange(1, board_size + 1, 1), rotation=90)  # Show all square numbers
plt.grid(axis='y')
plt.tight_layout()
plt.show()
