import numpy as np

def grid_score(grid: np.ndarray) -> float:
    score1 = 0 # player one's heuristic score
    score2 = 0 # player two's heuristic score
    # vvv temp values for counting diagonal scores (FIRST is for the diagonal from top left to bottom right, SCND is for the other diagonal)
    diag1FIRST = diag2FIRST = diag1SCND = diag2SCND = 0

    for i in range(3):
        # score by rows
        count1 = 0
        count2 = 0
        # count number of ones and twos into count1 and count2 respectively
        for j in range(3):
            val = grid[i][j]
            if val == 1:
                count1 += 1
            elif val == 2:
                count2 += 1
        # add counts to respective score if row is winnable by that player
        # if count1 == 3:
        #     return 1
        # elif count2 == 3:
        #     return -1
        if count2 == 0:
            score1 += count1
        elif count1 == 0:
            score2 += count2
        
        # repeat for columns
        count1 = 0
        count2 = 0
        for k in range(3):
            val = grid[k][i]
            if val == 1:
                count1 += 1
            elif val == 2:
                count2 += 1
        if count2 == 0:
            score1 += count1
        elif count1 == 0:
            score2 += count2

        # repeat for diagonals
        val1 = grid[i][i]
        val2 = grid[2-i][i]
        if val1 == 1:
            diag1FIRST += 1
        elif val1 == 2:
            diag2FIRST += 1
        if val2 == 1:
            diag1SCND += 1
        elif val2 == 2:
            diag2SCND += 1
    if diag2FIRST == 0:
        score1 += diag1FIRST
    elif diag1FIRST == 0:
        score2 += diag2FIRST
    if diag2SCND == 0:
        score1 += diag1SCND
    elif diag1SCND == 0:
        score2 += diag2SCND

    return score1 - score2

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
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]]) #-2
grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 2, 0]]) #-1

print(v2(grid3), sigmoid(v2(grid3)))
# trial(grid3)