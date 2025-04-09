import numpy as np
from utils import State, Action

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

# def getDepthFromZeros(zeros: int) -> int:
#     if zeros < 10:
#         return 100
#     if zeros < 14:
#         return 9
#     if zeros < 16:
#         return 8
#     if zeros < 20:
#         return 7
#     if zeros < 25:
#         return 6
#     if zeros < 40:
#         return 5
#     return 4

positionalScores = [
    [0.1, 0, 0.1],
    [0, 0.2, 0],
    [0.1, 0, 0.1],
]
sigmoidParam = 2.0
def localScores(grid: np.ndarray) -> float:
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

    print(score)
    return float(score)/sigmoidParam

grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
print(localScores(grid))

# # print(state.local_board_status)
# print(globalScore(state))
# # print(state.invert().local_board_status)
# print(globalScore(state.invert()))

# from utilsHJ import getLocalScores
# print(getLocalScores(state))