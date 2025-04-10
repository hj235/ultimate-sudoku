from utils import State, Action
import numpy as np
from scipy.special import expit
import pickle
import json
import hashlib

mem = {}

with open("stateCache.pkl", 'rb') as f:
    mem = dict(pickle.load(f))

def hashState(state:State):
    obj = (str(state.board), \
        state.prev_local_action is None or state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]])
    json_str = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

weights = [0.2018592438751993, -0.25221845757394545, 0.08584065936553029, 0.1897280963581898, -0.22438240628296782, 0.07625941758443595, 0.19454299489693902, -0.24191932505110864, 0.08380153720458879, 0.21983397901078575, -0.26462445214407276, 0.08797199919562305, 0.19549507425491713, -0.23548696645258466, 0.07960789896603584, 0.19741310270702087, -0.23992544460815807, 0.0809820808697964, 0.2090950751697677, -0.25028548220839614, 0.08635879520246315, 0.22584616245475805, -0.2704891262833827, 0.09103479839756336, -0.0013459372348249767, 0.036321962899336535, -0.024076746963873994, 0.6494277884913477]
intercept = 0.0005437624388256424

# def loadCache(fileName: str="stateCache.pkl"):
#     with open(fileName, 'rb') as f:
#         mem = pickle.load(f)

def saveCache(fileName: str="stateCache.pkl"):
    with open(fileName, 'wb') as f:
        pickle.dump(list(mem.items()), f)

def getGlobalFeatures(lines: list[list[float]], stepBypass: bool, fillNum: int, stepBypassAmplifier:float) -> tuple[list[float], float, float]:
    score = scoreSq = scoreCb = 0
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
        lineScore = lineScoreSq = lineScoreCb = 0
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

        score += lineScore
        scoreSq += lineScoreSq
        scoreCb += lineScoreCb
        i += 1

    bypassScore = 0
    if stepBypass and fillNum == 1:
        bypassScore = abs(score)*stepBypassAmplifier
        res.append(abs(score))
        res.append(abs(scoreSq))
        res.append(abs(scoreCb))
        res.append(1)
    elif stepBypass and fillNum == 2:
        bypassScore = -abs(score)*stepBypassAmplifier
        res.append(-abs(score))
        res.append(-abs(scoreSq))
        res.append(-abs(scoreCb))
        res.append(-1)
    else:
        bypassScore = 0
        res.append(0)
        res.append(0)
        res.append(0)
        res.append(0)
    return res, score, bypassScore

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

def getDepthFromZeros(zeros: int) -> int:
    if zeros < 20:
        return 5
    if zeros < 15:
        return 6
    if zeros < 10:
        return 100
    return 4

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

class StudentAgent:
    def __init__(self, depth=4, sigmoidParam=3.0, stepBypassAmplifier=1.15):
        """Instantiates your agent.
        """
        self.depth = depth
        self.sigmoidParam = sigmoidParam
        self.stepBypassAmplifier = stepBypassAmplifier

    # This function calculates a score for a local board, and assumes that the game within the local board has not yet ended.
    # The score is positive if player 1 is winning, negative if player 2 is winning, and 0 if the game is even.
    def localScores(self, grid: np.ndarray) -> float:
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
                score += ones*ones
            elif twos and not ones:
                score -= twos*twos
    
        return float(score)/self.sigmoidParam
    
    def globalScore(self, state: State):
        localBoardStatus = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                if state.local_board_status[i][j] == 0:
                    localBoardStatus[i][j] = expit(self.localScores(state.board[i][j]))
                else:
                    localBoardStatus[i][j] = state.local_board_status[i][j]
        localBoardLines = getLinesImm(localBoardStatus)

        stepByPass = state.prev_local_action and \
            state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
        features, heurScore, bypassScore = \
            getGlobalFeatures(localBoardLines, stepByPass, state.fill_num, self.stepBypassAmplifier)
        
        mlScore = 0
        for i in range(28):
            mlScore += features[i]*weights[i]
        mlScore += intercept

        score = ((expit(mlScore)-0.5)*2 + heurScore + bypassScore)/2
        return (expit(score)-0.5)*2
    
    def utility(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility()
        hsh = hashState(state)
        if hsh in mem:
            return mem[hsh]
        else:
            util = self.globalScore(state)
            mem[hsh] = util
            return util
    
    def minimax(self, state: State, depth: int, alpha:float, beta: float, zeroCount) -> Action:
        _, best_action = self.maximise(state, depth, alpha, beta, 0, zeroCount)
        return best_action
    
    # returns tuple of utility value of that state, and the action taken (None if terminal or depth reached)
    def maximise(self, state: State, depth: int, alpha: float, beta: float, stepBypasses: int, zeroCount: int) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if state.prev_local_action and state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0:
            stepBypasses += 1
        if depth == 0 or (stepBypasses >= 2 and zeroCount >= 35) or (stepBypasses >= 3 and zeroCount >= 28) or (stepBypasses >= 4 and zeroCount >= 20):
            return self.utility(state), None
        best_val = -np.inf
        best_action = None
        for action in state.get_all_valid_actions():
            copy = state.clone()
            new_state = copy.change_state(action)
            next_val, _ = self.minimise(new_state, depth - 1, alpha, beta, stepBypasses, zeroCount-1)
            if next_val > best_val:
                best_val = next_val
                best_action = action
            alpha = max(alpha, best_val)
            if best_val >= beta:
                return best_val, best_action
        return best_val, best_action
    
    def minimise(self, state: State, depth: int, alpha: float, beta: float, stepBypasses: int, zeroCount: int) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if state.prev_local_action and state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0:
            stepBypasses += 1
        if depth == 0 or (stepBypasses >= 2 and zeroCount >= 35) or (stepBypasses >= 3 and zeroCount >= 28) or (stepBypasses >= 4 and zeroCount >= 20):
            return self.utility(state), None
        best_val = np.inf
        best_action = None
        for action in state.get_all_valid_actions():
            copy = state.clone()
            new_state = copy.change_state(action)
            next_val, _ = self.maximise(new_state, depth - 1, alpha, beta, stepBypasses, zeroCount-1)
            if next_val < best_val:
                best_val = next_val
                best_action = action
            beta = min(beta, best_val)
            if best_val <= alpha:
                return best_val, best_action
        return best_val, best_action
        
    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        state = state.invert()
        # print("test")
        zeroCount = countZeros(state)
        depth = getDepthFromZeros(zeroCount)
        best_action = self.minimax(state, depth, -np.inf, np.inf, zeroCount)
        return best_action