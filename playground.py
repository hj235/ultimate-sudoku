import numpy as np
from utils import State, Action

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

# grid3 = np.array([[0, 0, 1], [0, 2, 0], [0, 0, 0]]) #-1
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]]) #1
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]]) #-2
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 2, 0]]) #-1

# Huhh
def countZerosDebug(state: State):
    count = 0
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] != 0:
                print(f'Local Board {i},{j} completed, skipping.')
                continue
            
            localCount  = 0
            grid = state.board[i][j]
            for r in range(3):
                for c in range(3):
                    if grid[r][c] == 0:
                        localCount += 1
            count += localCount
            print(f'localCount {localCount} at local board {i},{j}. globalCount {count}')
    return count


def countZeros(state: State):
    count = 0
    for i in range(3):
        for j in range(3):
            if state.local_board_status[i][j] != 0:
                continue
            
            grid = state.board[i][j]
            for r in range(3):
                for c in range(3):
                    if grid[r][c] == 0:
                        count += 1
    return count

def hashState(state: State):
    return hash((str(state.board), state.prev_local_action, state.fill_num))

# print(state)
# print(countZeros(state) == countZerosDebug(state))
# print(np.sum(board==0))
# state = State()
# print(np.sum(state.board == 0))

def getDepthFromZeros(zeros: int) -> int:
    if zeros < 10:
        return 100
    if zeros < 20:
        return 6
    if zeros < 40:
        return 5
    return 4

print(hashState(state))