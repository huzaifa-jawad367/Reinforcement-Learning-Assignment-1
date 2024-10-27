# Lets change the discount factor to 0.1 for a more myopic agent.

def value_iteration(states, actions, transition_probs, rewards, gamma=0.1, theta=1e-10):
    
    pass

    # Initialize value function
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for s in states:
            v = V[s]
            state_action_values = []

            # Calculate the value for each action
            for a in actions:
                action_value = 0
                for (next_state, prob) in transition_probs[s][a].items():
                    action_value += prob * (rewards[s][a][next_state] + gamma * V[next_state])
                state_action_values.append(action_value)

            V[s] = max(state_action_values)
            delta = max(delta, abs(v - V[s])) # Calculate the maximum change in value function

        # if the delta reaches the threshold convergence is reached
        if delta < theta:
            break

    # Calculate the optimal policy
    policy = {}
    for s in states:
        state_action_values = {}
        for a in actions:
            state_action_value = 0
            for (next_state, prob) in transition_probs[s][a].items():
                reward = rewards[s][a][next_state]
                state_action_value += prob * (reward + gamma * V[next_state])
            state_action_values[a] = state_action_value

        optimal_action = max(state_action_values, key=state_action_values.get)

        policy[s] = optimal_action

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

    # print("States:", states)
    # print("Actions:", actions)
    # print("\nTransition Probabilities:")
    # print(transition_probs)
    # print("\nRewards:")
    # print(rewards)

    V, policy = value_iteration(states, actions, transition_probs, rewards, gamma=0.1)

    print("\nOptimal Value Function:")
    for state in states:
        print(f"V({state}) = {V[state]:.6f}")

    print("\nOptimal Policy:")
    for state in states:
        print(f"π({state}) = {policy[state]}")
