# Reinforcement Learning Algorithms
For a Reinforcement Learning class, I worked on a few algorithms :

* Policy Iteration
* Value Iteration
* SARSA
* Q-Learning

to work on the OpenAI gym *Cliff Walking* problem (for SARSA and Q-Learning) and Sutton's Reinforcement Learning book *Grid World* exercice (for Policy Iteration and Value Iteration).

## How to use ?
`python main.py {RD, VI, PI, SARSA, QL}` With {RD: Random, VI: Value Iteration, PI: Policy Iteration, SARSA: SARSA, QL: Q-Learning}
Use `python main.py -h` to know more.

For Policy Iteration and Value Iteration, plots will appear, showing a map for each move (LEFT, RIGHT, UP, DOWN) colored when the given move is the best for the square.
For SARSA and Q-Learning, plots will appear, showing the final reward after each episode. The parameters have been tuned so that the learning works (reward increase along the episodes).
