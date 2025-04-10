from stateMapping import StudentAgent as Cacher
from stateMapping import saveCache
from submissionV5 import StudentAgent
from utils import State, Action
import time

class RandomStudentAgent(StudentAgent):
    def choose_action(self, state: State) -> Action:
        # If you're using an existing Player 1 agent, you may need to invert the state
        # to have it play as Player 2. Uncomment the next line to invert the state.
        # state = state.invert()

        # Choose a random valid action from the current game state
        return state.get_random_valid_action()

def run(your_agent: StudentAgent, opponent_agent: StudentAgent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0
    
    state = State(fill_num=start_num)
    
    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone())
        end_time = time.time()
        
        random_action = state.get_random_valid_action()
        if end_time - start_time > 3:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)

    print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")
        
    if state.terminal_utility() == 1:
        print("You win!")
    elif state.terminal_utility() == 0:
        print("You lose!")
    else:
        print("Draw")

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")
        
    print(f"Turn count: {turn_count}\n")

your_agent = lambda: StudentAgent()
opponent_agent = lambda: Cacher()

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

saveCache()