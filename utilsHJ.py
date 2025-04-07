import numpy as np
from utils import State, Action
from scipy.special import expit

sigmoidParam=2.0

# get every row, column and diagonal
def getLines(grid: np.ndarray) -> list:
    lines = [[]]*8

    for i in range(3):
        lines[i] = grid[i]
        lines[i+3] = grid.T[i]
    lines[6] = [0]*3
    lines[7] = [0]*3
    for i in range(3):
        lines[6][i] = grid[i][i]
        lines[7][i] = grid[2-i][i]
    return lines

# get every row, column and diagonal, not referencing original grid
def getLinesImm(grid: np.ndarray) -> list:
    lines = [[0]*3 for _ in range(8)]

    for i in range(3):
        for j in range(3):
            lines[i][j] = grid[i][j]
            lines[i+3][j] = grid.T[i][j]
    for i in range(3):
        lines[6][i] = grid[i][i]
        lines[7][i] = grid[2-i][i]
    return lines

def getLocalScores(state: State) -> np.ndarray:
    res = np.zeros(9, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            res[i*3+j] = expit(localScore(state.board[i][j])) if state.local_board_status[i][j] == 0 \
                               else state.local_board_status[i][j]
    return res

# This function calculates a score for a local board, and assumes that the game within the local board has not yet ended.
# The score is positive if player 1 is winning, negative if player 2 is winning, and 0 if the game is even.
def localScore(grid: np.ndarray) -> float:
    lines = getLines(grid)

    # calculate score from potential wins
    score = 0
    for line in lines:
        ones = 0
        twos = 0
        for val in line:
            if val == 1:
                ones += 1
            elif val == 2:
                twos += 1
        if ones and not twos:
            score += ones
        elif twos and not ones:
            score -= twos

    return float(score)/sigmoidParam