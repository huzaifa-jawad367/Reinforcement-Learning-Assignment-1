# Assignment 1 - Value Iteration and Policy Iteration Algorithms

In this assignment we were given the task of solving a small MDP pertaining to a little autonomous rover on Mars. We had to implement the value iteration and policy iteration algorithms for the given MDP.


## Task 1

Here is the Markov Decision Process created for the Mars Rover.


<img width="595" alt="Screenshot 2024-10-27 at 11 32 13 PM" src="https://github.com/user-attachments/assets/8946c003-b635-4971-a47d-3aec5befa4d2">

Here is a clearer image created through the use of python script and `NetworkX` library. The Bidirectional arrows are equivalent to 2 arrows going on back and forth.

![Figure_1](https://github.com/user-attachments/assets/4134c083-37c3-47cc-afa3-cf201a2882e4)

## Task 2

<img width="726" alt="Screenshot 2024-10-27 at 10 13 08 PM" src="https://github.com/user-attachments/assets/667aae4c-62e9-4499-8906-a42966af752c">

This shows the expected return the agent i.e the mars rover would get. The highest reward is expected when starting at the Bottom state followed by the top state and then rolling down.
Values reflect the rewards and the transition probabilities defined in the environment, showing how each state accumulates reward over time.

## Task 3
<img width="856" alt="Screenshot 2024-10-27 at 10 36 55 PM" src="https://github.com/user-attachments/assets/29487c42-47ce-470e-aacf-d7e092adc0dd">
<img width="868" alt="Screenshot 2024-10-27 at 10 53 43 PM" src="https://github.com/user-attachments/assets/0972c7b3-b1a3-4737-a36e-ccc8d348d343">

## Task 4
<img width="842" alt="Screenshot 2024-10-27 at 11 00 19 PM" src="https://github.com/user-attachments/assets/25744ea8-8637-48fa-bbcc-54f7e719d799">
Here i changed the discount factor from 0.9 to 0.1 therefore giving a more myopic evaluation the policy chaged to always be driving as it always looked for maximising the return value in the short term. What changed from the original implementation was that it drove in rolling down state as well instead of coming to the bottom and trying again to drive from the bottom state hence maximizing reward.


<img width="926" alt="Screenshot 2024-10-27 at 11 09 21 PM" src="https://github.com/user-attachments/assets/e62b7fe0-6d78-477b-9ce8-0b41ba20370b">
Here I increase the probabilities for the more desired outcome i.e staying at the top. I will change the transition probability in the rolling down state 

```'drive': [('top', 0.6, 2), ('rolling_down', 0.2, 1.5), ('bottom', 0.2, 0.5)],```

it is more attractive for the agent to stay at the top due to higher reward and is more likely to stay on top due to transition probabilities

<img width="841" alt="Screenshot 2024-10-27 at 11 17 51 PM" src="https://github.com/user-attachments/assets/772c5fcf-1ff9-4eb1-b784-712fc39ea1fd">

In this image we changed the reward at the bottom state transitioning to the bottom while driving. Hence the policy shows that it is ideal to drve at the top and it maximises the reward and when rolling down, not to drive so as to get to the bottom again and then drive again as staying at the bottom maximises the reward.
