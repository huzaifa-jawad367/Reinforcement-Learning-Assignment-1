import numpy as np

def policy_evaluation_with_deteministic_policy(policy, states, actions, trans_probs, rewards, gamma=0.9):
    V = np.zeros(len(states))
    state_indices = {s: i for i, s in enumerate(states)}
    A = np.zeros((len(states), len(states)))
    b = np.zeros(len(states))

    for s in states:
        idx = state_indices[s]
        A[idx, idx] = 1
        action = policy[s]
        for next_state in trans_probs[s][action]:
            prob = trans_probs[s][action][next_state]
            reward = rewards[s][action][next_state]
            j = state_indices[next_state]
            b[idx] += prob * reward
            A[idx, j] = A[idx, j] - gamma * prob

    V = np.linalg.solve(A, b)
    V_dict = {state: V[state_indices[state]] for state in states}
    return V_dict   

def policy_iteration_with_deteministic_policy(states, actions, trans_probs, rewards, gamma=0.9):
    policy = {state: "don't drive" for state in states}
    
    stable_policy = False
    while not stable_policy:
        # Policy evaluation
        V = policy_evaluation_with_deteministic_policy(policy, states, actions, trans_probs, rewards, gamma)
        # Policy improvement
        stable_policy = True
        for s in states:
            old_action = policy[s]
            state_action_values = {}
            for a in actions:
                state_action_value = 0
                for next_state in trans_probs[s][a]:
                    prob = trans_probs[s][a][next_state]
                    reward = rewards[s][a][next_state]
                    state_action_value += prob * (reward + gamma * V[next_state])
                state_action_values[a] = state_action_value
            
            best_action = max(state_action_values, key=state_action_values.get)
            policy[s] = best_action
            if old_action != best_action:
                stable_policy = False
        
    return V, policy
    

if __name__ == '__main__':

    states = ['top', 'rolling_down', 'bottom']
    actions = ['drive', "don't drive"]

    # Representing the MDP using a dictionary
    transitions = {
        'top': {
            'drive': [('top', 0.5, 2), ('rolling_down', 0.5, 2)],
            "don't drive": [('top', 0.5, 3), ('rolling_down', 0.5, 1)]
        },
        'rolling_down': {
            'drive': [('top', 0.3, 2), ('rolling_down', 0.4, 1.5), ('bottom', 0.3, 0.5)],
            "don't drive": [('bottom', 1.0, 1)]
        },
        'bottom': {
            'drive': [('top', 0.5, 2), ('bottom', 0.5, 2)],
            "don't drive": [('bottom', 1.0, 1)]
        }
    }

    # Separate dictionaries for transition probabilities and rewards
    transition_probs = {}  # Structure: {state: {action: {next_state: probability}}}
    rewards = {}           # Structure: {state: {action: {next_state: reward}}}

    for state, actions_dict in transitions.items():
        transition_probs[state] = {}
        rewards[state] = {}
        for action, outcomes in actions_dict.items():
            transition_probs[state][action] = {}
            rewards[state][action] = {}
            for outcome in outcomes:
                next_state, prob, reward = outcome
                transition_probs[state][action][next_state] = prob
                rewards[state][action][next_state] = reward

    V, policy = policy_iteration_with_deteministic_policy(states, actions, transition_probs, rewards)

    print("Values:")
    print(V)

    print("\nOptimal Policy:")
    for state in states:
        print(f"State: {state}, Action: {policy[state]}")