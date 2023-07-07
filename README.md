# Snake_RL
Project for the 2022-23 Reinforcement Learning project at Units.
Implementation of the video game "Snake" solved through RL techniques.  

More information on the theoretical aspects can be found in `report.pdf`.

## How to run

To play the game snake run:

```bash
python Game.py
```

The speed of the game can be modified by changing the constant `SPEED`.  

To train a model for `S` steps, save the Q-value matrix to `model`
and save an image and numpy array of the scores as `name`,
using an `algorithm` among _SARSA, expected SARSA, Q-learning_, run:
```bash
python train_driver.py -f model -s S -n name -a algorithm
```

To make a `model` (as a `.npy` file) play the game, run:
```bash
python main.py model width height
```
where `width` and `height` are optional arguments that change the size of the board.

## Content

- `Game.py` human playable implementation of the game.  
- `AIGame.py` Implementation of the game made to be auto-played.
- `train_driver.py` trains a model, showing progres and saving.
- `Agent.py` Implementation of the agent that takes actions based on the states.
The decision process is decided by the model that can be trained.
- `model.py` has the class that implements temporal difference control.
- `data/` contains the trained models, a graph of their training along with the arrays used for the plots.  