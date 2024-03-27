import numpy as np

# Define the size of the board and create an empty transition matrix
board_size = 100
P = np.zeros((board_size, board_size))

# Define a simplified version of chutes (snakes) and ladders as {start: end}
# In a real game, you would fill this in with the actual positions of chutes and ladders
chutes_and_ladders = {
    1: 38,  # Ladder from 1 to 38
    4: 14,  # Ladder from 4 to 14
    9: 31,  # Ladder from 9 to 31
    16: 6,  # Chute from 16 to 6
    21: 42,  # Ladder from 21 to 42
    36: 44,  # Ladder from 36 to 44
    47: 26,  # Chute from 47 to 26
    49: 11,  # Chute from 49 to 11
    56: 53,  # Chute from 56 to 53
    62: 19,  # Chute from 62 to 19
    64: 60,  # Chute from 64 to 60
    71: 91,  # Ladder from 71 to 91
    80: 100, # Ladder from 80 to 100
    87: 24,  # Chute from 87 to 24
    93: 73,  # Chute from 93 to 73
    95: 75,  # Chute from 95 to 75
    98: 78,  # Chute from 98 to 78
}

# Populate the transition matrix
for i in range(board_size):
    for roll in range(1, 7):  # Possible die rolls
        if i + roll <= board_size:
            # Check if the square leads to a chute or ladder
            if i + roll in chutes_and_ladders:
                # Move directly to the chute or ladder's destination
                P[i, chutes_and_ladders[i + roll] - 1] += 1/6
            else:
                # Normal move
                P[i, i + roll] += 1/6

# Correct for the last six squares where you can roll too high
for i in range(95, 100):
    P[i, 95:] = P[95, 95:]

# The game is over when you reach the last square, so it's an absorbing state
P[99, 99] = 1

# Show a small sample of the matrix to check correctness
P[:10, :10]