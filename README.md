# Reinforcement Learning Tic-Tac-Toe

Although Tic-Tac-Toe is a solved game that can easily be played by non-AI agents (search trees, etc.). Writing a perfectly playing AI is straightforward using these methods. However, Tic-Tac-Toe also provides a good opportunity to be a first attempt for Reinforcement Learning, Q-Learning, and Deep Q-Learning, which is what I do in this project.


## Markov Decision Process (MDP)

For Q-Learning, we have to model the Tic-Tac-Toe environment as a Markov Decision Process `MDP = <S, A, T, R>`. Player 1 is the agent doing the learning, and player 2 is either playing randomly or as a human.

### States (S)
The game state is represented by an array `[None, 1, 2, None, None, None, None, None, 1]` where 1 represents the first player, 2 represents the second player, and None represents an empty space. The state space is all the different boards where it is the first player's (AI's) turn to play.

### Actions (A)
Given a game board, the legal actions are the open spaces on the board.

### Transition Function (T)
The transition function is the probability that given an old state S and an action A, it will end up at state S'. Since player 2 plays randomly, there is an equal probability of reaching all S' from S given A.

### Reward (R)
After transitioning from state S to S', a reward is given to the agent. For a winning state, a reward of 100 is given; for a losing state, a penalty of -100; for a tie state, a reward of 50; and all other intermediate states, a reward of 0. 

With this distribution of rewards, winning is preferable to tieing, which is preferable to losing.

### Value
The value is the reward that the agent expects to get in the long run while at state S under policy &pi;

### Policy
An agent will have an optimal policy &pi;* for each game state that optimizes the rewards it can win in the future. It will choose the action that correlates with the largest expected tuture reward.


## Q-Learning
Now that we have the Tic-Tac-Toe environment set up, we can have it start playing games against a random opponent to try to learn the sequences of actions that maximize its reward. 

For simple games like Tic-Tac-Toe, it's possible to keep the policy in what's called a Q-table. This Q-table contains all possible game states and the expected rewards for each action, A. As the game complexity increases, it becomes infeasible to store every game state in a Q-table. It's at this point that you have to move to Deep Q-Learning and train a Neural Network to learn the optimal Q values.

I trained two separate agents: one where the AI always goes first, and one where the first player is randomly chosen. Due to the nature of Tic-Tac-Toe, the first player has an advantage over the second player, winning many more games in completely random play. The state space for AI first is also much lower ~3000 than random first ~6000, resulting in varying training times.

### Updating Q-table
During random play, the agent will occasionally end up in states that given rewards or penalties. To provide learning for the future, the appropriate value in the Q-table needs to be updated to show the expected reward at that state. The equation to update the Q-table is below:

`Q(s)[a] := ((1 - α) * Q(s)[a]) + α * (r + (γ * max a′Q(s′)[a′]))`

Where Q(s) is the old state, Q(s') is the new state, alpha is the learning rate, and gamma is the discount factor. The `max a'Q(s')[a']` section says select highest expected reward from S' over all possible actions A.

The next state definition was something that took a little bit to figure out. While the agent was able to choose winning moves, it wasn't able to figure out how to block states that lead to wins by the opponent. What eventually worked was making the next state S' be the game board after player 2 moved randomly. This way, the agent was able to figure out which states and actions lead to defeat (and how to make actions to avoid these states).

## Exploration vs. Exploitation
While learning during random play, the agent has the ability to either play randomly or use the learned Q-table to choose intelligent moves that maximize the expected reward. Playing randomly allows the agent to explore and discover new game states, while exploiting allows the agent to use know winning sequences. 

This is done by using a parameter called epsilon. As training goes on, the value of epsilon is slowly decreased, and the agent begins choosing more intelligent moves and less random moves.

`[0, -42.05936354914919, 0, 99.99999998941072, 0, 62.09929903388406, 0, 0, 0]` is an example of a value in the Q-table given a state. It represents the nine spaces of the game board. By playing a number of games that included this state, it learned that playing in the 4th position always gives a win, while playing in the 2nd space usually resulted in a loss. During actual play, the agent chooses the action with the highest value, in this case, space four.


## Q-Learning Results
The AI first agent was able to reach perfect play (no losses) after ~150,000 training games.
Alternatively, the random first agent took ~300,000 games to reach perfect play. This is due to the fact that the state space is twice as large as the AI first state space. 

While other methods are faster and more efficient than Q-learning, this still shows that agents can obtain a high (perfect) level of play without any human/strategic input.