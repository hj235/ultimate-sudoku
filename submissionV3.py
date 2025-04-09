from utils import State, Action
import numpy as np
from scipy.special import expit

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

def getDepthFromZeros(zeros: int) -> int:
    if zeros >= 36:
        return 4
    elif zeros >= 22:
        return 5
    elif zeros >= 18:
        return 6
    elif zeros >= 15:
        return 7
    elif zeros >= 13:
        return 8
    else:
        return 100

def hashState(state: State):
    return hash((str(state.board), state.prev_local_action, state.fill_num))

class StudentAgent:
    mem = {}

    def __init__(self, depth=4, sigmoidParam=3.0, stepBypassAmplifier=1.15):
        """Instantiates your agent.
        """
        self.depth = depth
        self.sigmoidParam = sigmoidParam
        self.stepBypassAmplifier = stepBypassAmplifier

        # predict some initial states
        state = State()
        self.choose_action(state)
        state = state.invert()
        self.choose_action(state)

    def cacheStateUtil(self, state: State, util: float) -> None:
        h = hashState(state)
        StudentAgent.mem[h] = util

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
                score += (ones + zeros)*(ones + zeros)
            elif twos and not ones and not threes:
                score -= (twos + zerosTwo)*(twos + zerosTwo)
    
        # amplify score if step rule is bypassed
        last_local_action = state.prev_local_action
        step_bypass_bonus = 0
        if state.local_board_status[last_local_action[0]][last_local_action[1]] != 0:
            step_bypass_bonus = abs(score*self.stepBypassAmplifier)
        if state.fill_num == 1:
            score += step_bypass_bonus
        elif state.fill_num == 2:
            score -= step_bypass_bonus
        
        return score
    
    def utility(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility()
        else:
            if state in StudentAgent.mem:
                return StudentAgent.mem[state]
            else:
                util = self.globalScore(state)
                self.cacheStateUtil(state, util)
                return util
    
    def minimax(self, state: State, depth: int, alpha:float, beta: float) -> Action:
        _, best_action = self.maximise(state, depth, alpha, beta, 0)
        return best_action
    
    # returns tuple of utility value of that state, and the action taken (None if terminal or depth reached)
    def maximise(self, state: State, depth: int, alpha: float, beta: float, parent_util: float) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if depth == 0:
            return parent_util, None
        best_val = -np.inf
        best_action = None

        # sort actions according to utility of resulting state
        actions = state.get_all_valid_actions()
        n_actions = len(actions)
        infoList: list[tuple[State, Action, float]] = [None]*n_actions
        for i in range(n_actions):
            a = actions[i]
            copy = state.clone()
            s = copy.change_state(a)
            u = self.utility(s)
            infoList[i] = (a, s, u)
        infoList.sort(key=lambda info: info[2], reverse=True)

        # find best action with pruning
        for action, new_state, new_util in infoList:
            next_val, _ = self.minimise(new_state, depth - 1, alpha, beta, new_util)
            if next_val > best_val:
                best_val = next_val
                best_action = action
            alpha = max(alpha, best_val)
            if best_val >= beta:
                return best_val, best_action
        return best_val, best_action
    
    def minimise(self, state: State, depth: int, alpha: float, beta: float, parent_util: float) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if depth == 0:
            return parent_util, None
        best_val = np.inf
        best_action = None

        # sort actions according to utility of resulting state
        actions = state.get_all_valid_actions()
        n_actions = len(actions)
        infoList: list[tuple[State, Action, float]] = [None]*n_actions
        for i in range(n_actions):
            a = actions[i]
            copy = state.clone()
            s = copy.change_state(a)
            u = self.utility(s)
            infoList[i] = (a, s, u)
        infoList.sort(key=lambda info: info[2])

        # find best action with pruning
        for action, new_state, new_util in infoList:
            next_val, _ = self.maximise(new_state, depth - 1, alpha, beta, new_util)
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
        depth = getDepthFromZeros(countZeros(state))
        best_action = self.minimax(state, depth, -np.inf, np.inf)
        return best_action