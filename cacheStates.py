import pickle
import json
import hashlib
from utils import load_data, State, Action
from submissionV5 import StudentAgent

data = load_data()

def hashState(state:State):
    obj = (str(state.board), \
        state.prev_local_action is None or state.local_board_status[state.prev_local_action[0]][state.prev_local_action[1]])
    json_str = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

agent = StudentAgent()
mem = {}

for state, _ in data:
    hsh = hashState(state)
    util = agent.utility(state)
    mem[hsh] = util

with open("stateCache.pkl", 'wb') as f:
    pickle.dump(list(mem.items()), f)
