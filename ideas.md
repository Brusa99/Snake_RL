# HIGHSCORES

- 60 (brusa)
- 54 (giovanni)

# Slides

## Introduction

Describe the game:
- how you move (discrete actions)  
- when do you fail (hit wall/itself)  
- goal: eat the apple (you grow)
- terminal state? all the grid is covered by the snake

## Formalization

what are states?  
the state is the board, so the states are made of board configuration with empty spaces, reward, block.  Is it enough? no. The snake must know the direction it is travelling in (?) head is different than block.  

why didn't we choose other states?  
If the snake knows the previous board configuration it knows the direction but this double the cost in memory DOUBLES INPUT SIZE (also more neurons in the network likely to be required)  
If we restrict the vision the snake might not find the food=reward.  

## Rewards

food +1  
game over -1  
normal travel $-\epsilon$ (to avoid loops)

## What algorithm do we use

Q learning?  
With Neural Networks  

### Problems

In the late game most of the board is blocked.
With an $\epsilon$-greedy policy, a random action is taken.
In the late game this will most likely lead to death so the game will nearly never be completed.


