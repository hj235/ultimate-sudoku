import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from utils import Action, State


def check_board(board):
    cnt_twos_p1 = 0
    cnt_twos_p2 = 0
    cnt_lines_p1 = 0
    cnt_lines_p2 = 0
    cnt_p1_blocks = 0
    cnt_p2_blocks = 0
    pos1 = 0
    pos2 = 0

    for i in range(3):
        # check rows
        if board[i, 0] == board[i, 1]:
            if board[i, 0] == 1:
                cnt_twos_p1 += 1
                if board[i, 2] == 2:
                    cnt_p2_blocks -= 1
            elif board[i, 0] == 2:
                cnt_twos_p2 -= 1
                if board[i, 2] == 1:
                    cnt_p1_blocks += 1
            elif board[i, 0] == 0:
                if board[i, 2] == 1:
                    cnt_lines_p1 += 1
                elif board[i, 2] == 2:
                    cnt_lines_p2 -= 1

        elif board[i, 0] == board[i, 2]:
            if board[i, 0] == 1:
                cnt_twos_p1 += 1
                if board[i, 1] == 2:
                    cnt_p2_blocks -= 1
            elif board[i, 0] == 2:
                cnt_twos_p2 -= 1
                if board[i, 1] == 1:
                    cnt_p1_blocks += 1
            elif board[i, 0] == 0:
                if board[i, 1] == 1:
                    cnt_lines_p1 += 1
                elif board[i, 1] == 2:
                    cnt_lines_p2 -= 1

        elif board[i, 1] == board[i, 2]:
            if board[i, 1] == 1:
                cnt_twos_p1 += 1
                if board[i, 0] == 2:
                    cnt_p2_blocks -= 1
            elif board[i, 1] == 2:
                cnt_twos_p2 -= 1
                if board[i, 0] == 1:
                    cnt_p1_blocks += 1
            elif board[i, 1] == 0:
                if board[i, 0] == 1:
                    cnt_lines_p1 += 1
                elif board[i, 0] == 2:
                    cnt_lines_p2 -= 1

        # check columns
        if board[0, i] == board[1, i]:
            if board[0, i] == 1:
                cnt_twos_p1 += 1
                if board[2, i] == 2:
                    cnt_p2_blocks -= 1
            elif board[0, i] == 2:
                cnt_twos_p2 -= 1
                if board[2, i] == 1:
                    cnt_p1_blocks += 1
            elif board[0, i] == 0:
                if board[2, i] == 1:
                    cnt_lines_p1 += 1
                elif board[2, i] == 2:
                    cnt_lines_p2 -= 1

        elif board[0, i] == board[2, i]:
            if board[0, i] == 1:
                cnt_twos_p1 += 1
                if board[1, i] == 2:
                    cnt_p2_blocks -= 1
            elif board[0, i] == 2:
                cnt_twos_p2 -= 1
                if board[1, i] == 1:
                    cnt_p1_blocks += 1
            elif board[0, i] == 0:
                if board[1, i] == 1:
                    cnt_lines_p1 += 1
                elif board[1, i] == 2:
                    cnt_lines_p2 -= 1

        elif board[1, i] == board[2, i]:
            if board[1, i] == 1:
                cnt_twos_p1 += 1
                if board[0, i] == 2:
                    cnt_p2_blocks -= 1
            elif board[1, i] == 2:
                cnt_twos_p2 -= 1
                if board[0, i] == 1:
                    cnt_p1_blocks += 1
            elif board[1, i] == 0:
                if board[0, i] == 1:
                    cnt_lines_p1 += 1
                elif board[0, i] == 2:
                    cnt_lines_p2 -= 1

        for j in range(3):
            if board[i, j] == 1:
                pos1 += StudentAgent.WEIGHTS[i, j]
            elif board[i, j] == 2:
                pos2 -= StudentAgent.WEIGHTS[i, j]

    # check diagonals
    if board[0, 0] == board[1, 1]:
        if board[0, 0] == 1:
            cnt_twos_p1 += 1
            if board[2, 2] == 2:
                cnt_p2_blocks -= 1
        elif board[0, 0] == 2:
            cnt_twos_p2 -= 1
            if board[2, 2] == 1:
                cnt_p1_blocks += 1
        elif board[0, 0] == 0:
            if board[2, 2] == 1:
                cnt_lines_p1 += 1
            elif board[2, 2] == 2:
                cnt_lines_p2 -= 1

    elif board[0, 0] == board[2, 2]:
        if board[0, 0] == 1:
            cnt_twos_p1 += 1
            if board[1, 1] == 2:
                cnt_p2_blocks -= 1
        elif board[0, 0] == 2:
            cnt_twos_p2 -= 1
            if board[1, 1] == 1:
                cnt_p1_blocks += 1
        elif board[0, 0] == 0:
            if board[1, 1] == 1:
                cnt_lines_p1 += 1
            elif board[1, 1] == 2:
                cnt_lines_p2 -= 1

    elif board[1, 1] == board[2, 2]:
        if board[1, 1] == 1:
            cnt_twos_p1 += 1
            if board[0, 0] == 2:
                cnt_p2_blocks -= 1
        elif board[1, 1] == 2:
            cnt_twos_p2 -= 1
            if board[0, 0] == 1:
                cnt_p1_blocks += 1
        elif board[1, 1] == 0:
            if board[0, 0] == 1:
                cnt_lines_p1 += 1
            elif board[0, 0] == 2:
                cnt_lines_p2 -= 1

    if board[0, 2] == board[1, 1]:
        if board[0, 2] == 1:
            cnt_twos_p1 += 1
            if board[2, 0] == 2:
                cnt_p2_blocks -= 1
        elif board[0, 2] == 2:
            cnt_twos_p2 -= 1
            if board[2, 0] == 1:
                cnt_p1_blocks += 1
        elif board[0, 2] == 0:
            if board[2, 0] == 1:
                cnt_lines_p1 += 1
            elif board[2, 0] == 2:
                cnt_lines_p2 -= 1

    elif board[0, 2] == board[2, 0]:
        if board[0, 2] == 1:
            cnt_twos_p1 += 1
            if board[1, 1] == 2:
                cnt_p2_blocks -= 1
        elif board[0, 2] == 2:
            cnt_twos_p2 -= 1
            if board[1, 1] == 1:
                cnt_p1_blocks += 1
        elif board[0, 2] == 0:
            if board[1, 1] == 1:
                cnt_lines_p1 += 1
            elif board[1, 1] == 2:
                cnt_lines_p2 -= 1

    elif board[1, 1] == board[2, 0]:
        if board[1, 1] == 1:
            cnt_twos_p1 += 1
            if board[0, 2] == 2:
                cnt_p2_blocks -= 1
        elif board[1, 1] == 2:
            cnt_twos_p2 -= 1
            if board[0, 2] == 1:
                cnt_p1_blocks += 1
        elif board[1, 1] == 0:
            if board[0, 2] == 1:
                cnt_lines_p1 += 1
            elif board[0, 2] == 2:
                cnt_lines_p2 -= 1

    # return cnt_twos_p1, cnt_twos_p2, cnt_p1_blocks, cnt_p2_blocks, cnt_lines_p1, cnt_lines_p2, pos
    return (
        cnt_twos_p1,
        cnt_twos_p2,
        cnt_p1_blocks,
        cnt_p2_blocks,
        cnt_lines_p1,
        cnt_lines_p2,
        pos1,
        pos2,
    )
    # return  cnt_p1_blocks, cnt_p2_blocks, cnt_lines_p1, cnt_lines_p2, pos


class StudentAgent:
    WEIGHTS = np.array(
        [
            [3.0, 2.0, 3.0],
            [2.0, 4.0, 2.0],
            [3.0, 2.0, 3.0],
        ]
    )

    def __init__(self, depth: int = 4, is_max=True):
        """Instantiates your agent."""
        self.max_depth = depth
        self.is_max = is_max
        self.model = make_pipeline(PolynomialFeatures(1), LinearRegression())
        dummy_input = np.zeros((1, 17))
        self.model.fit(dummy_input, np.zeros((1, 1)))
        self.model.named_steps["linearregression"].coef_ = np.array(
            [
                0.0,
                -0.5909663089945503,
                0.5966476557253176,
                -0.07613957201647148,
                0.06303782864619235,
                -0.37394714664726275,
                0.37396673743799025,
                0.4356117809206966,
                -0.0005735957848945303,
                0.010711754852845477,
                0.03710583095892225,
                -0.06445357820038504,
                -0.025622788705228675,
                0.02819685527633619,
                0.025169063902273678,
                0.4503109726930108,
            ]
        )
        self.model.named_steps["linearregression"].intercept_ = -0.003883036521635755

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        if not self.is_max:
            state = state.invert()

        _, move = self._minimax(
            alpha=-np.inf,
            beta=np.inf,
            depth=self.max_depth,
            state=state,
            is_max=True,
        )

        return move

    def _minimax(self, alpha, beta, depth, state: State, is_max):
        if state.is_terminal():
            score = state.terminal_utility()

            if score == 1:  # we win
                return 9999999, None
            if score == 2:  # they win
                return -9999999, None
            return 0, None  # no win

        if depth <= 0:
            return self._heuristic(state), None

        moves = state.get_all_valid_actions()

        move = moves[0]

        if is_max:
            next_state = state.change_state(move)
            value, _ = self._minimax(alpha, beta, depth - 1, next_state, False)
            alpha = max(alpha, value)

            for action in moves[1:]:
                if alpha >= beta:
                    break

                new_state = state.change_state(action)
                tmp_v, _ = self._minimax(alpha, beta, depth - 1, new_state, False)

                if tmp_v > value:
                    value = tmp_v
                    move = action

                alpha = max(alpha, value)
        else:
            next_state = state.change_state(move)
            value, _ = self._minimax(alpha, beta, depth - 1, next_state, True)
            beta = min(beta, value)

            for action in moves[1:]:
                if alpha >= beta:
                    break

                new_state = state.change_state(action)
                tmp_v, _ = self._minimax(alpha, beta, depth - 1, new_state, True)

                if tmp_v < value:
                    value = tmp_v
                    move = action

                beta = min(beta, value)

        return value, move

    def _prep_feat(self, state: State) -> np.ndarray:
        global_stat = state.local_board_status
        send_val = 0

        if state.prev_local_action:
            if global_stat[state.prev_local_action] != 0:
                send_val = 1 if state.fill_num == 1 else -1
            feat = np.concatenate(
                (
                    check_board(global_stat),
                    check_board(state.board[state.prev_local_action]),
                    [send_val],
                )
            )
        else:
            feat = np.concatenate(
                (
                    check_board(global_stat),
                    np.zeros(shape=8),
                    [1 if state.fill_num == 1 else -1],
                )
            )
        return feat

    def _heuristic(self, state: State) -> int:
        feat = self._prep_feat(state)

        return self.model.predict(feat.reshape(1, -1))[0]
