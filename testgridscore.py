import numpy as np
from utils import State, Action

# def grid_score(grid: np.ndarray) -> float:
#     score1 = 0 # player one's heuristic score
#     score2 = 0 # player two's heuristic score
#     # vvv temp values for counting diagonal scores (FIRST is for the diagonal from top left to bottom right, SCND is for the other diagonal)
#     diag1FIRST = diag2FIRST = diag1SCND = diag2SCND = 0

#     for i in range(3):
#         # score by rows
#         count1 = 0
#         count2 = 0
#         # count number of ones and twos into count1 and count2 respectively
#         for j in range(3):
#             val = grid[i][j]
#             if val == 1:
#                 count1 += 1
#             elif val == 2:
#                 count2 += 1
#         # add counts to respective score if row is winnable by that player
#         # if count1 == 3:
#         #     return 1
#         # elif count2 == 3:
#         #     return -1
#         if count2 == 0:
#             score1 += count1
#         elif count1 == 0:
#             score2 += count2
        
#         # repeat for columns
#         count1 = 0
#         count2 = 0
#         for k in range(3):
#             val = grid[k][i]
#             if val == 1:
#                 count1 += 1
#             elif val == 2:
#                 count2 += 1
#         if count2 == 0:
#             score1 += count1
#         elif count1 == 0:
#             score2 += count2

#         # repeat for diagonals
#         val1 = grid[i][i]
#         val2 = grid[2-i][i]
#         if val1 == 1:
#             diag1FIRST += 1
#         elif val1 == 2:
#             diag2FIRST += 1
#         if val2 == 1:
#             diag1SCND += 1
#         elif val2 == 2:
#             diag2SCND += 1
#     if diag2FIRST == 0:
#         score1 += diag1FIRST
#     elif diag1FIRST == 0:
#         score2 += diag2FIRST
#     if diag2SCND == 0:
#         score1 += diag1SCND
#     elif diag1SCND == 0:
#         score2 += diag2SCND

#     return score1 - score2
# def grid_score(grid: np.ndarray) -> float:
#     score1 = 0 # player one's heuristic score
#     score2 = 0 # player two's heuristic score
#     # vvv temp values for counting diagonal scores (FIRST is for the diagonal from top left to bottom right, SCND is for the other diagonal)
#     diag0FCOUNT = diag0FIRST = diag1FIRST = diag2FIRST = diag3FIRST \
#         = diag0SCOUNT = diag0SCND = diag1SCND = diag2SCND = diag3SCND = 0

#     for i in range(3):
#         # score by rows
#         count0 = sum0 = count1 = count2 = count3 = 0
#         # count number of ones and twos into count1 and count2 respectively
#         for j in range(3):
#             val = grid[i][j]
#             if val == 1:
#                 count1 += 1
#             elif val == 2:
#                 count2 += 1
#             elif val ==3:
#                 count3 += 1
#             else:
#                 sum0 += val
#                 count0 += 1
#         if count2 == 0 and count3 == 0:
#             score1 += count1 + sum0
#         elif count1 == 0 and count3 == 0:
#             score2 += count2 + count0 - sum0
        
#         # repeat for columns
#         count0 = sum0 = count1 = count2 = count3 = 0
#         for k in range(3):
#             val = grid[k][i]
#             if val == 1:
#                 count1 += 1
#             elif val == 2:
#                 count2 += 1
#             elif val == 3:
#                 count3 += 1
#             else:
#                 count0 += val
#         if count2 == 0 and count3 == 0:
#             score1 += count1 + sum0
#         elif count1 == 0 and count3 == 0:
#             score2 += count2 + count0 - sum0

#         # repeat for diagonals
#         val1 = grid[i][i]
#         val2 = grid[2-i][i]
#         if val1 == 1:
#             diag1FIRST += 1
#         elif val1 == 2:
#             diag2FIRST += 1
#         elif val1 == 3:
#             diag3FIRST += 1
#         else:
#             diag0FIRST += val1
#             diag0FCOUNT += 1
#         if val2 == 1:
#             diag1SCND += 1
#         elif val2 == 2:
#             diag2SCND += 1
#         elif val2 == 3:
#             diag3SCND += 1
#         else:
#             diag0SCND += val2
#             diag0SCOUNT += 1
#     if diag2FIRST == 0 and diag3FIRST == 0:
#         score1 += diag1FIRST + diag0FIRST
#     elif diag1FIRST == 0 and diag3FIRST == 0:
#         score2 += diag2FIRST + diag0FIRST - diag0FCOUNT
#     if diag2SCND == 0 and diag3SCND == 0:
#         score1 += diag1SCND + diag0SCND
#     elif diag1SCND == 0 and diag3SCND == 0:
#         score2 += diag2SCND + diag0SCND - diag0SCOUNT

#     return score1 - score2

def trial(grid):
    print(grid)
    print(grid_score(grid))

def v2(grid: np.ndarray) -> float:
    # concatenate all rows(horizontal, vertical, diagonal) into a single array
    allRows = np.concatenate(( grid, grid.T, \
    grid.diagonal()[np.newaxis, :], np.fliplr(grid).diagonal()[np.newaxis, :]), \
    axis=0)
    
    ones, twos, zeros = (allRows == 1), (allRows == 2), (allRows == 0)
    onesOrZeros = np.logical_or(ones, zeros)
    twosOrZeros = np.logical_or(twos, zeros)
    oneCount = np.sum(ones, axis=1)
    twoCount = np.sum(twos, axis=1)
    oneScore = np.sum(oneCount[np.all(onesOrZeros, axis=1)])
    twoScore = np.sum(twoCount[np.all(twosOrZeros, axis=1)])
    return oneScore - twoScore

def sigmoid(x:float) -> float:
    return 1 / (1 + np.exp(-x))

# grid0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# grid = np.array([[2, 1, 1], [1, 2, 2], [1, 2, 1]])
# grid2 = np.array([[0, 0, 2], [0, 1, 0], [0, 0, 0]])

# grid3 = np.array([[0, 0, 1], [0, 2, 0], [0, 0, 0]]) #-1
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]]) #1
grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]]) #-2
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 2, 0]]) #-1

# grid4 = np.array([[0.5, 0.5, 1], [0.6, 2, 0.6], [1, 2, 0.3]]) #-1

board=np.array([
        [
            [[1, 0, 2], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 2, 0], [0, 0, 0], [0, 0, 0]],
        ],
    ])
state = State(
    board=board,
    fill_num=1,
    prev_action=(2, 2, 0, 1),
)
# print(grid4)
# print(grid4 != 1)
# arr = [[]]*8

# arr[1] = [1,2,3]
# arr.append([4,5,6])
# print(arr)
# print(grid_score(grid4))
# print(v2(grid4))

# trial(grid3)

positionalScores = np.array([
    [1, 0, 1],
    [0, 2, 0],
    [1, 0, 1],
])

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

# This function calculates a score for a local board, and assumes that the game within the local board has not yet ended.
# The score is positive if player 1 is winning, negative if player 2 is winning, and 0 if the game is even.
def v4(grid: np.ndarray) -> float:
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
    
    # reward for positional advantage
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 1:
                score += positionalScores[i][j]
            elif grid[i][j] == 2:
                score -= positionalScores[i][j]

    return score

def invert(grid):
    res = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 1:
                res[i][j] = 2
            elif grid[i][j] == 2:
                res[i][j] = 1
    return res

# print(grid3)
# print(v4(grid3))
# grid3inv = invert(grid3)
# print(grid3inv)
# print(v4(grid3inv))


# This function calculates a score for a local board, and assumes that the game within the local board has not yet ended.
# The score is positive if player 1 is winning, negative if player 2 is winning, and 0 if the game is even.
def v5(grid: np.ndarray) -> float:
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

    return float(score)/2.0

# print(grid3)
# print(v5(grid3))
# grid3inv = invert(grid3)
# print(grid3inv)
# print(v5(grid3inv))

class Laze:
    def __init__(self, value):
        self._value = value
        self._evaluated = False

    def get(self):
        if not self._evaluated:
            self._value = self._value()
            self._evaluated = True
        return self._value

from scipy.special import expit
# bounded to [-1, 1]
def boundedSigmoid(x) -> float:
    return expit(x)*2 - 1

def globalScore(state: State):
    localBoardStatus = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] == 0:
                localBoardStatus[i][j] = expit(v5(state.board[i][j]))
            else:
                localBoardStatus[i][j] = state.local_board_status[i][j]
    localBoardLines = getLinesImm(localBoardStatus)
    print(localBoardStatus)
    score = 0
    for line in localBoardLines:
        zeros = zerosTwo = ones = twos = threes = 0
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
        if ones and not twos and not threes:
            score += ones + zeros
        elif twos and not ones and not threes:
            score -= twos + zerosTwo
        # print(f'{line} {ones} {twos} {threes} {zeros}')
    
    return score

# print(state.local_board_status)
print(globalScore(state))
# print(state.invert().local_board_status)
print(globalScore(state.invert()))
