import numpy as np

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

# grid3 = np.array([[0, 0, 1], [0, 2, 0], [0, 0, 0]]) #-1
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 0]]) #1
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 0, 2]]) #-2
# grid3 = np.array([[0, 0, 1], [0, 2, 0], [1, 2, 0]]) #-1

# def getLines(grid: np.ndarray) -> list:
#     lines = [0]*8

print(np.sum(np.all(np.logical_or(board == 1, board == 0), axis=3), axis=1))
# print(np.sum(np.all(np.logical_or(board == 1, board == 0), axis=2), axis=1))
ones = 1
twos = 1
if ones and not twos:
    print("A")
elif twos and not ones:
    print("B")

# ========================= LAZY EVALUATION =========================
# gen = ((lambda : print(i)) for i in range(3)) 
# print(gen[1])