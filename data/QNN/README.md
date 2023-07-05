# LinearQNet Models

In this directory the results of the models trained with `self.model` set to an instance of `LinearQNet`.  
Different configurations of _actions_, _rewards_, value of $\epsilon_t$, _short/long_ memory updates, are tried.  

In particular:  
- `eps099` is the standard (3 actions, 11 states, $\epsilon_t = 0.99 \epsilon_{t-1}$, both lm and sm)
- `4valmat` has all the matrix given as state, with 4 possible actions.  
- `deponsteps` has $\epsilon_t = \epsilon_0 - text{number of games}$.  
- `increasedpens` has negative reward increased by a factor of 100.  
- `nolm` has only short memory training.  
- `crd` ??

## No longer supported

This models won't run with the current code since it no longer supports the model framework. Principal differences are:  
- The NN model outputs an array instead of a scalar.  
- The trainer is separate from the model.

#### Convergence Speed 

The run is executed only one time, so the graphs should be used to determine speed of convergence.

