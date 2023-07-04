# Snake_RL
Project for the 2022-23 Reinforcement Learning project at Units.
Implementation of the video game "Snake" solved through RL techniques.  

## How to run

To play the game snake run:

```bash
python Game.py
```

To train a model for `S` steps, save it to `model`
and save an image of the scores as `image`, run:
```bash
python train_driver.py -f model -s S -n image
```

## Content

- `Game.py` human playable implementation of the game.  
- `AIGame.py` Implementation of the game made to be auto-played.
- `Agent.py` Implementation of the agent that takes actions based on the states.
The decision process is decided by the model.
The training phase is handled by the provided trainer.  
- `data/` contains the trained models along with a graph of their score  
-- `deponsteps` is trained with epsilon0 = 80, epsilon_t = epsilon_0 - # of games  
-- `eps099` is trained with epsilon0 = 80, at each iteration epsilon <- 0.99 * epsilon  
-- `increasedpen` is trained with negative rewards multiplied by 1003