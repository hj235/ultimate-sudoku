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

actionsToLines = {
    (0,0): [0,3,6],
    (0,1): [0,4],
    (0,2): [0,5,7],
    (1,0): [1,3],
    (1,1): [1,4,6,7],
    (1,2): [1,5],
    (2,0): [2,3,7],
    (2,1): [2,4],
    (2,2): [2,5,6]
}

def getGlobalFeatures(lines: list[list[float]], stepBypass: bool, fillNum: int, prevAction: tuple) -> list[float]:
    score = 0
    scoreSq = 0
    scoreCb = 0
    actionScore = 0
    actionScoreSq = 0
    actionScoreCb = 0
    res = []
    i = 0
    for line in lines:
        zeros = zerosTwo = zerosHalf = ones = twos = threes = 0
        for val in line:
            if val == 1:
                ones += 1
            elif val == 2:
                twos += 1
            elif val == 3:
                threes += 1
            else:
                zeros += val
                zerosTwo += (1-val)
                zerosHalf += (val-0.5)
        lineScore = 0
        lineScoreSq = 0
        lineScoreCb = 0
        if ones and not twos and not threes:
            lineScore = (ones + zeros)
            lineScoreSq = (ones + zeros)*(ones + zeros)
            lineScoreCb = lineScoreSq*(ones + zeros)
        elif twos and not ones and not threes:
            lineScore = -(twos + zerosTwo)
            lineScoreSq = -(twos + zerosTwo)*(twos + zerosTwo)
            lineScoreCb = lineScoreSq*(twos + zerosTwo)
        elif not ones and not twos and not threes:
            lineScore = zerosHalf
            lineScoreSq = zerosHalf*abs(zerosHalf)
            lineScoreCb = lineScoreSq*abs(zerosHalf)
        else:
            lineScore = 0
            lineScoreSq = 0
            lineScoreCb = 0
        res.append(lineScore)
        res.append(lineScoreSq)
        res.append(lineScoreCb)
        if prevAction is not None and i in actionsToLines[prevAction]:
            actionScore += lineScore
            actionScoreSq += lineScoreSq
            actionScoreCb += lineScoreCb
        score += lineScore
        scoreSq += lineScoreSq
        scoreCb += lineScoreCb
        i += 1

    if stepBypass and fillNum == 1:
        res.append(abs(score))
        res.append(abs(scoreSq))
        res.append(abs(scoreCb))
        res.append(1)
    elif stepBypass and fillNum == 2:
        res.append(-abs(score))
        res.append(-abs(scoreSq))
        res.append(-abs(scoreCb))
        res.append(-1)
    # elif fillNum == 1:
    #     res.append(abs(actionScore))
    #     res.append(abs(actionScoreSq))
    #     res.append(0)
    # elif fillNum == 2:
    #     res.append(-abs(actionScore))
    #     res.append(-abs(actionScoreSq))
    #     res.append(0)
    else:
        res.append(0)
        res.append(0)
        res.append(0)
        res.append(0)

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