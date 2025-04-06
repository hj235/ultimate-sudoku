from utils import State, Action
import numpy as np

class StudentAgent:
    def __init__(self, depth=2):
        """Instantiates your agent.
        """
        self.depth = depth
        self.weight = 10 # IF HAVE TIME USE ML FOR THIS WEIGHT
    
    def sigmoid(self, x:float) -> float:
        return 1 / (1 + np.exp(-x)) 
    
    # grid score calculation algorithm:
    #   for every possible row (horizontal, vertical & diagonal, 8 total)
    #       if player 1 can win
    #           add number of player 1's tiles to score1
    #       if player 2 can win
    #           add number of player 2's tiles to score2
    # return sigmoid(score1 - score2)
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
    def utility(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility()
        else:
            return self.computeGlobalScore(state)

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        best_action, best_value = None, -np.inf
        actions = state.get_all_valid_actions()
        for action in actions:
            val = self.utility(state.change_state(action))
            if val > best_value:
                best_value = val
                best_action = action
        if best_action is None:
            return state.get_random_valid_action()
        return best_action