import pickle
from utils import State, load_data
from utilsHJ import getLocalScores
import numpy as np

def tfmData(data: list[tuple[State, float]]) -> list[tuple[list[float], float]]:
    res = []
    for state, utility in data:
        res.append((getLocalScores(state), utility))
    return res

def saveTransformedData(data: list[tuple[State, float]]):
    transformed = tfmData(data)
    with open('tfmData.pkl', 'wb') as f:
        pickle.dump(transformed, f)

def loadData(filename) -> list[tuple[list[float], float]]:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

data = load_data()
saveTransformedData(data)