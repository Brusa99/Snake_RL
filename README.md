# Snake_RL
Project for the 2022-23 Reinforcement Learning project at Units.
Implementation of the video game "Snake" solved through RL techniques.  

## How to run

To play the game snake run:

```bash
python Game.py
```

To train a model for `S` steps, save it to `model`
and save an image of the scores as `image`,
using an `algorithm` among _SARSA, expected SARSA, Q-learning_, run:
```bash
python train_driver.py -f model -s S -n image -a algorithm
```

## Content

- `Game.py` human playable implementation of the game.  
- `AIGame.py` Implementation of the game made to be auto-played.
- `Agent.py` Implementation of the agent that takes actions based on the states.
The decision process is decided by the model.
The training phase is handled by the provided trainer.  
- `data/` contains the trained models, a graph of their training along with the arrays used for the plot  