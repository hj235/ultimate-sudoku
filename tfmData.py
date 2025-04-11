import pickle
from utils import State, load_data
from utilsHJ import getLocalScores, getLinesImm, getGlobalFeatures
import numpy as np

def tfmData(data: list[tuple[State, float]]) -> list[tuple[list[float], float]]:
    res = []
    for state, utility in data:
        res.append((getLocalScores(state), utility))
    return res

# def saveTransformedData(data: list[tuple[State, float]]):
#     transformed = tfmData(data)
#     with open('tfmData.pkl', 'wb') as f:
#         pickle.dump(transformed, f)

# def saveTransformedDataV2():
#     data = loadData("tfmData.pkl")
#     res = []
#     for vals, utility in data:
#         grid = np.array(vals).reshape((3, 3))
#         linesImm = getLinesImm(grid)
#         res.append((linesImm, utility))
    
#     with open("tfmData2.pkl", "wb") as wf:
#         pickle.dump(res, wf)

# def saveTransformedDataV3():
#     data = loadData("tfmData2.pkl")
#     res = []
#     for lines, utility in data:
#         res.append((getGlobalLineScores(lines), utility))
    
#     with open("tfmData3.pkl", "wb") as wf:
#         pickle.dump(res, wf)

# def saveTransformedDataV4():
#     data = loadData("tfmData3.pkl")
#     res = []
#     for scores, utility in data:
#         res.append((scores, (utility+1.0)/2.0))
    
#     with open("tfmData4.pkl", "wb") as wf:
#         pickle.dump(res, wf)

def saveTransformedDataV5():
    data = load_data()
    res = []
    for state, utility in data:
        if utility == 1 or utility == -1:
            continue
        localScores = getLocalScores(state)
        linesImm = getLinesImm(np.array(localScores).reshape((3,3)))
        stepBypass = False
        if state.prev_local_action is None or \
            state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0:
            stepBypass = True
        # print(state.prev_local_action)
        # if state.prev_local_action:
            # print(state.local_board_status)
        features = getGlobalFeatures(linesImm, stepBypass, state.fill_num, state.prev_local_action)
        newUtil = 0 if utility/2 + 0.5 == 0 else -np.log(1/(utility/2 + 0.5) - 1)
        res.append((features, newUtil))

    with open('tfmData18.pkl', 'wb') as wf:
        pickle.dump(res, wf)

def loadData(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

# saveTransformedDataV5()