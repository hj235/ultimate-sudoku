from utils import State, Action
import numpy as np
from scipy.special import expit

weights = [0.6810941477220519, -0.9042635796537578, 0.3134889477979552, 0.6092802191020059, -0.739385726206488, 0.263217462253697, 0.6136453812842638, -0.8118858971471219, 0.2929948080578667, 0.6817359031192158, -0.916861423204613, 0.31661831210473146, 0.7029397018866511, -0.9365706672837834, 0.3208877412374687, 0.6139810712337012, -0.8447671631426145, 0.2997030494596738, 0.6454522997688666, -0.8676621747524681, 0.31578535205664005, 0.6466364464292942, -0.8944423641589636, 0.3234671776516659, 0.15173314884363184, -0.12526697114106897, -0.01707946825890002, 3.458081557151088]
intercept = 0.006091123617134075

def getGlobalFeatures(lines: list[list[float]], stepBypass: bool, fillNum: int) -> list[float]:
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
    else:
        res.append(0)
        res.append(0)
        res.append(0)
        res.append(0)
    return res

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
        features = getGlobalFeatures(localBoardLines, stepByPass, state.fill_num)
        score = 0
        for i in range(28):
            score += features[i]*weights[i]
        score += intercept
    
        return score
    
    def utility(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility()
        else:
            return self.globalScore(state)
    
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
        # state = state.invert()
        # print("test")
        zeroCount = countZeros(state)
        depth = getDepthFromZeros(zeroCount)
        best_action = self.minimax(state, depth, -np.inf, np.inf, zeroCount)
        return best_action