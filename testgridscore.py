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
        val1, val2 = grid[i][i], grid[2-i][i]
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
    elif diag1FIRST == 0:
        score2 += diag2SCND

    return score1 - score2

def trial(grid):
    print(grid)
    print(grid_score(grid))

grid1 = np.array([[0, 0, 2], [0, 1, 0], [0, 0, 0]])
trial(grid1)