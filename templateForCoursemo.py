from utils import State, Action
import numpy as np

class StudentAgent:
    def __init__(self, depth=4):
        """Instantiates your agent.
        """
        self.depth = depth
        self.weight = 10 # IF HAVE TIME USE ML FOR THIS WEIGHT
    
    def sigmoid(self, x:float) -> float:
        return 1 / (1 + np.exp(-x)) 
    
    # grid score calculation algorithm:
    #   for every possible row (horizontal, vertical & diagonal, 8 total)
    #       if player 1 cannot win +0 to score1 and continue
    #       if player 1 can win
    #           add number of player 1's tiles to score1
    #       if player 2 cannot win +0 to score2 and continue
    #       if player 2 can win
    #           add number of player 2's tiles to score2
    # return score1 - score2
    def computeLocalScore(self, grid: np.ndarray) -> float:
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
        return self.sigmoid(oneScore - twoScore)
    
    def computeGlobalScore(self, state: State) -> float:
        local_scores = state.local_board_status.astype(np.float64)
        for r in range(3):
            for c in range(3):
                if local_scores[r][c] == 0:
                    local_scores[r][c] = self.computeLocalScore(state.board[r][c])
        
        allRows = np.concatenate((local_scores, local_scores.T, \
        local_scores.diagonal()[np.newaxis, :], np.fliplr(local_scores).diagonal()[np.newaxis, :]), \
        axis=0)

        notTwoOrThree = np.logical_and(allRows != 2, allRows != 3)
        notOneOrThree = np.logical_and(allRows != 1, allRows != 3)
        sums = np.sum(allRows, axis = 1)
        oneScore = np.sum(sums[np.all(notTwoOrThree, axis=1)])
        twoScore = np.sum(sums[np.all(notOneOrThree, axis=1)])
        return self.sigmoid(oneScore - twoScore)

    # returns the utility value of the state (use sigmoid function for ML?)
    # maybe not, so that i can reuse this func for global? or create a local_grid_score
    def utility(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility()
        else:
            return self.computeGlobalScore(state)
    
    def minimax(self, state: State, depth: int, alpha:float, beta: float) -> Action:
        _, best_action = self.maximise(state, depth, alpha, beta)
        return best_action
    
    # returns tuple of utility value of that state, and the action taken (None if terminal or depth reached)
    def maximise(self, state: State, depth: int, alpha: float, beta: float) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if depth == 0:
            return self.utility(state), None
        best_val = -np.inf
        best_action = None
        for action in state.get_all_valid_actions():
            copy = state.clone()
            new_state = copy.change_state(action)
            next_val, _ = self.minimise(new_state, depth - 1, alpha, beta)
            if next_val > best_val:
                best_val = next_val
                best_action = action
            alpha = max(alpha, best_val)
            if best_val >= beta:
                return best_val, best_action
        return best_val, best_action
    
    def minimise(self, state: State, depth: int, alpha: float, beta: float) -> tuple:
        if state.is_terminal():
            return state.terminal_utility(), None
        if depth == 0:
            return self.utility(state), None
        best_val = np.inf
        best_action = None
        for action in state.get_all_valid_actions():
            copy = state.clone()
            new_state = copy.change_state(action)
            next_val, _ = self.maximise(new_state, depth - 1, alpha, beta)
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
        best_action = self.minimax(state, self.depth, -np.inf, np.inf)
        return best_action