import numpy as np

# Define the size of the board and create an empty transition matrix
board_size = 100
P = np.zeros((board_size, board_size))

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

# Show a small sample of the matrix to check correctness
print(P[:10, :10])