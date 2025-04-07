import numpy as np
from sklearn.linear_model import LinearRegression
from utils import State, Action
import time

class StudentAgent:
    def __init__(self):
        """Instantiates your agent.
        """

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        def lm(state):
            model = LinearRegression()
            model.coef_ = np.array([1.1422312942e-03, -1.1462350435e-02, -5.5590867817e-03, 1.0110752091e-02,
                                    3.3921375646e-03, 6.8687136617e-04, 4.9474707971e-03, -5.0718836600e-03,
                                    -5.0644872503e-03, 6.5805346477e-03, 2.8156611161e-03, 4.2663345839e-04,
                                    -4.2717330761e-03, 1.2808172084e-02, -7.4956786584e-03, -1.0824923794e-03,
                                    6.3141339452e-04, 1.1457198165e-03, -3.2221441259e-03, 3.2590808185e-04,
                                    -7.2740029187e-05, -3.4618749941e-03, -2.3301556676e-03, 5.9288198747e-03,
                                    6.1905176199e-03, 1.4916472824e-03, -1.8648743622e-03, 2.2696230338e-03,
                                    -3.8452836116e-03, -1.8684944480e-03, -1.4676407278e-03, 1.2879150038e-02,
                                    4.7378295945e-03, 3.8513282510e-03, -6.6713301396e-03, 2.1047625843e-03,
                                    -5.3892369346e-03, -3.5676475648e-03, -9.7986437307e-03, 5.7013321330e-03,
                                    -1.4189387923e-02, 7.8153736929e-04, -8.0928187189e-03, -9.3381478583e-03,
                                    -9.4229751127e-03, 5.3704816574e-03, -2.6403512587e-03, 4.9481685404e-03,
                                    9.8036289472e-04, 9.3822398646e-03, 1.3041074694e-03, 3.9116328311e-03,
                                    3.7388933762e-03, 2.7327839057e-03, 4.0566877178e-05, -3.5907377035e-03,
                                    -1.0622134773e-02, -9.6710612729e-04, -1.2420654637e-02, -9.7007549408e-03,
                                    7.8139689485e-03, 1.2287888754e-04, 7.9430090944e-03, -3.1610413614e-03,
                                    -2.3001953212e-03, 4.8645125855e-03, 9.3274799573e-03, 1.6358599736e-02,
                                    -9.8984555353e-03, 6.9424110382e-03, 2.6862697451e-04, 2.5538270617e-04,
                                    -7.2543003444e-03, -3.9652499355e-03, -3.6683839949e-04, 4.9508438560e-04,
                                    6.4762421389e-03, 6.7457659463e-03, 7.6007869957e-04, -2.0932787460e-03,
                                    -4.7524543598e-03]
                                )
            model.intercept_ = 0.0004950875000013586
            return np.dot(state.board.flatten(), model.coef_) + model.intercept_
            
        EMPTY = 0
        PLAYER_ONE = 1
        PLAYER_TWO = 2
        DRAW = 3

        POSITION_WEIGHTS = [
            [1.5, 1.0, 1.5],
            [1.0, 2.0, 1.0],  
            [1.5, 1.0, 1.5]   
        ]

        WIN_PATTERNS = [
            # Rows
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            # Columns
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            # Diagonals
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)]
        ]

        def combined_eval(state):
            # Get scores from both evaluation methods
            heuristic_score = eval(state)
            lr_score = lm(state)
            
            # Determine the game stage (early, mid, late)
            move_count = count_total_moves(state)
            early_game = move_count < 20
            late_game = move_count > 50
            
            if early_game:
                # In early game, favor heuristic evaluation which has good strategic understanding
                combined_score = 0.8 * (heuristic_score + 100) + 0.2 * lr_score * 10000
            elif late_game:
                # In late game, lean more on the learned model which may have better pattern recognition
                combined_score = 0.3 * (heuristic_score + 100) + 0.7 * lr_score * 10000
            else:
                # In mid-game, balance both approaches
                combined_score = 0.5 * (heuristic_score + 100) + 0.5 * lr_score * 10000

            return combined_score

        def eval(state):
            score = evaluate_board_status(state.local_board_status)
            score += detect_global_threats(state.local_board_status)

            global_control_score = evaluate_global_control_patterns(state.local_board_status)
            score += global_control_score

            for i in range(3):
                for j in range(3):
                    local_status = state.local_board_status[i][j]
                    board_position_weight = POSITION_WEIGHTS[i][j]
                    
                    # If local board is still in play, evaluate its state
                    if local_status == EMPTY:
                        local_score = evaluate_board(state.board[i][j])
                        score += local_score * board_position_weight * 1.2
                        
                        # Bonus if this is where the opponent must play next
                        if state.prev_local_action == (i, j):
                            # If this is a good board for us, it's better to send opponent here
                            score += local_score * 0.4
                    
                    # Add value for won local boards
                    elif local_status == PLAYER_ONE:
                        score += 4 * board_position_weight
                    elif local_status == PLAYER_TWO:
                        score -= 4 * board_position_weight
                    elif local_status == DRAW:
                        score -= 2 * board_position_weight
            
            if state.local_board_status[1][1] == PLAYER_ONE:
                score += 7
            elif state.local_board_status[1][1] == PLAYER_TWO:
                score -= 7
            elif state.local_board_status[1][1] == DRAW:
                score -= 1

            if state.prev_local_action is None:
                score += 3 if state.fill_num == PLAYER_ONE else -3
            if state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0:  # Can play anywhere
                score += 3 if state.fill_num == PLAYER_ONE else -3
            else:
                # Count valid moves in the next local board
                valid_moves = count_valid_moves(state, state.prev_local_action)
                if valid_moves == 0:  # Board is full, can play anywhere
                    score += 2 if state.fill_num == PLAYER_ONE else -2
                else:
                    # Having more valid moves in the destination board is good
                    move_flexibility = valid_moves / 9  # Normalize to [0,1]
                    score += move_flexibility * 2 if state.fill_num == PLAYER_ONE else -move_flexibility * 2

            move_count = count_total_moves(state)
            game_stage_factor = min(1.0, move_count / 40)
            
            if game_stage_factor > 0.7:
                score *= (1 + 0.6 * game_stage_factor)
            else:
                score *= (1 + 0.5 * game_stage_factor)
            
            # Consider forcing opponent to suboptimal boards
            if state.prev_local_action:
                next_i, next_j = state.prev_local_action
                if state.local_board_status[next_i][next_j] == EMPTY:
                    local_board_value = evaluate_board(state.board[next_i][next_j])
                    # Negative because we sent opponent to this board
                    score += -local_board_value * 0.9
            
            return score
        
        def evaluate_board(board):
            score = 0
            for i in range(3):
                for j in range(3):
                    if board[i][j] == PLAYER_ONE:
                        score += POSITION_WEIGHTS[i][j]
                    elif board[i][j] == PLAYER_TWO:
                        score -= POSITION_WEIGHTS[i][j]

            # Detect win threats
            score += detect_win_threats(board)
            
            # Detect fork opportunities (two threats at once)
            score += detect_fork_opportunities(board)
            
            return score
        
        def evaluate_board_status(status):
            score = 0

            for i in range(3):
                for j in range(3):
                    if status[i][j] == PLAYER_ONE:
                        score += POSITION_WEIGHTS[i][j] * 3
                    elif status[i][j] == PLAYER_TWO:
                        score -= POSITION_WEIGHTS[i][j] * 3
    
            # Check for win threats at the global level
            score += detect_win_threats(status)
            
            # Detect fork opportunities at global level
            score += detect_fork_opportunities(status)
            
            return score
        
        def detect_global_threats(status):
            threats = 0
            for pattern in WIN_PATTERNS:
                p1 = sum(1 for i, j in pattern if status[i][j] == PLAYER_ONE)
                empty = sum(1 for i, j in pattern if status[i][j] == EMPTY)
                if p1 == 2 and empty == 1:
                    threats += 1
            return threats
        
        def evaluate_global_control_patterns(status):
            score = 0
            
            # Evaluate diagonal control (important for strategic play)
            diag1 = [(0, 0), (1, 1), (2, 2)]
            diag2 = [(0, 2), (1, 1), (2, 0)]
            
            # Count control of first diagonal
            p1_control = sum(1 for i, j in diag1 if status[i][j] == PLAYER_ONE)
            p2_control = sum(1 for i, j in diag1 if status[i][j] == PLAYER_TWO)
            empty = sum(1 for i, j in diag1 if status[i][j] == EMPTY)
            
            if p1_control > 0 and p2_control == 0:
                score += p1_control * 5
            if p2_control > 0 and p1_control == 0:
                score -= p2_control * 5
            
            # Count control of second diagonal
            p1_control = sum(1 for i, j in diag2 if status[i][j] == PLAYER_ONE)
            p2_control = sum(1 for i, j in diag2 if status[i][j] == PLAYER_TWO)
            empty = sum(1 for i, j in diag2 if status[i][j] == EMPTY)
            
            if p1_control > 0 and p2_control == 0:
                score += p1_control * 5
            if p2_control > 0 and p1_control == 0:
                score -= p2_control * 5
            
            # Evaluate control of corners (strategic advantage)
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            p1_corners = sum(1 for i, j in corners if status[i][j] == PLAYER_ONE)
            p2_corners = sum(1 for i, j in corners if status[i][j] == PLAYER_TWO)
            
            score += p1_corners * 3
            score -= p2_corners * 3
            
            return score
        
        def detect_win_threats(board):
            score = 0
    
            for pattern in WIN_PATTERNS:
                # Extract values for this pattern
                values = [board[i][j] for i, j in pattern]
                
                # Count player pieces and empty spaces
                p1_count = values.count(PLAYER_ONE)
                p2_count = values.count(PLAYER_TWO)
                empty_count = values.count(EMPTY)
                
                # Two pieces with one empty = win threat
                if p1_count == 2 and empty_count == 1:
                    score += 4  # Player 1 has a win threat
                if p2_count == 2 and empty_count == 1:
                    score -= 4  # Player 2 has a win threat
                    
                # One piece with two empty = potential future threat
                if p1_count == 1 and empty_count == 2:
                    score += 2  # Player 1 has a developing threat
                if p2_count == 1 and empty_count == 2:
                    score -= 2  # Player 2 has a developing threat
            
            return score
        
        def detect_fork_opportunities(board):
            score = 0
            
            # Check for each empty cell
            for i in range(3):
                for j in range(3):
                    if board[i][j] == EMPTY:
                        # Count potential win threats created by placing a piece here
                        p1_threats = count_win_threats_at_position(board, i, j, PLAYER_ONE)
                        p2_threats = count_win_threats_at_position(board, i, j, PLAYER_TWO)
                        
                        # Two or more threats is a fork
                        if p1_threats >= 2:
                            score += 4  # Player 1 has a fork opportunity
                        if p2_threats >= 2:
                            score -= 4  # Player 2 has a fork opportunity
            
            return score
        
        def count_win_threats_at_position(board, i, j, player):
            # Make a hypothetical move
            board_copy = [row[:] for row in board]
            board_copy[i][j] = player
            
            # Count threats
            threats = 0
            for pattern in WIN_PATTERNS:
                if (i, j) in pattern:  # Only check patterns that include this position
                    values = [board_copy[x][y] for x, y in pattern]
                    player_count = values.count(player)
                    empty_count = values.count(EMPTY)
                    
                    # Two pieces with one empty = win threat
                    if player_count == 2 and empty_count == 1:
                        threats += 1
                        
            return threats
        
        def count_valid_moves(state, local_board):
            i, j = local_board
            count = 0
            for x in range(3):
                for y in range(3):
                    if state.board[i][j][x][y] == EMPTY:
                        count += 1
            return count
        
        def count_total_moves(state):
            count = 0
            for i in range(3):
                for j in range(3):
                    for x in range(3):
                        for y in range(3):
                            if state.board[i][j][x][y] != EMPTY:
                                count += 1
            return count

        MAX_DEPTH = 6
        TIME_LIMIT = 2.8


        def is_timeout(start):
            return time.time() - start >= TIME_LIMIT
        
        def order_moves(state, actions, maximizing=True):
            action_scores = []
            for action in actions:
                next_state = state.clone().change_state(action)
                score = quick_eval(next_state, action)
                action_scores.append((action, score))
            
            # Sort by score (descending for maximizer, ascending for minimizer)
            action_scores.sort(key=lambda x: x[1], reverse=maximizing)
            
            # Return just the sorted actions
            return [a[0] for a in action_scores]
        
        def quick_eval(state, action):
            # Check if this move wins a local board
            local_i, local_j, pos_i, pos_j = action
            board = state.board[local_i][local_j]
            
            # Check if this move creates a win threat on local board
            score = 0
            
            # Prioritize center positions
            score += POSITION_WEIGHTS[pos_i][pos_j] * 2
            
            # Prioritize moves that send opponent to disadvantageous boards
            next_local_board = (pos_i, pos_j)
            if state.local_board_status[next_local_board[0]][next_local_board[1]] != EMPTY:
                score += 10  # Free move advantage
            else:
                # Check if next board is almost full (limited options)
                valid_moves = count_valid_moves(state, next_local_board)
                if valid_moves <= 2:
                    score += 5
            
            return score
        
        def max_value(state, alpha, beta, depth, max_d, start):
            if is_timeout(start):
                return combined_eval(state)
            if state.is_terminal():
                return state.terminal_utility()
            if depth >= max_d:
                return eval(state)
            v = float("-inf")
            actions = state.get_all_valid_actions()
            # actions = order_moves(state, actions)
            # actions.sort(key=lambda action: eval(state.clone().change_state(action)))
            for action in actions:
                v = max(v, min_value(state.clone().change_state(action), alpha, beta, depth + 1, max_d, start))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
                if is_timeout(start):
                    break
            return v
        def min_value(state, alpha, beta, depth, max_d, start):
            if is_timeout(start):
                return combined_eval(state)
            if state.is_terminal():
                return state.terminal_utility()
            if depth >= max_d:
                return eval(state)
            v = float("inf")
            actions = state.get_all_valid_actions()
            # actions = order_moves(state, actions, maximizing=False)
            # actions.sort(key=lambda action: eval(state.clone().change_state(action)), reverse=True)
            for action in actions:
                v = min(v, max_value(state.clone().change_state(action), alpha, beta, depth + 1, max_d, start))
                if v <= alpha:
                    return v
                beta = min(beta, v)
                if is_timeout(start):
                    break
            return v
        def minimax(state):
            start = time.time()
            best_action = None
            best_value = float("-inf")
            for curr_depth in range(1, MAX_DEPTH + 1):
                if is_timeout(start):
                    break
                curr_best_value = float("-inf")
                curr_best_action = None
                alpha = float("-inf")
                beta = float("inf")
                actions = state.get_all_valid_actions()
                # actions = order_moves(state, actions)
                for action in actions:
                    if is_timeout(start):
                        break
                        
                    next_state = state.clone().change_state(action)
                    value = min_value(next_state, alpha, beta, 1, curr_depth, start)
                    
                    if value > curr_best_value:
                        curr_best_value = value
                        curr_best_action = action
                    alpha = max(alpha, value)
                if not is_timeout(start):
                    best_action = curr_best_action
                    best_value = curr_best_value
            return best_action
        
        
            # states = state.get_all_valid_actions()
            # states.sort(key=lambda action: eval(state.clone().change_state(action)), reverse=True)
            # for action in states:
            #     value = max_value(state.clone().change_state(action), float("-inf"), float("inf"), 0, start)
            #     if value > best_value:
            #         best_value = value
            #         best_action = action
            # return best_action
        return minimax(state)