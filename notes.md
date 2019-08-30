# Questions
* Why do you need to pass `itr` to `agent.eval_mode()` etc.? Is this for agents that are using, e.g., annealed epsilon-greedy strategies? What about when you just want to load a fully-trained agent and use it? What value of `itr` should be passed? `eval_mode` should perhaps have whatever that value is as a default.

# Custom Environment
* How do you set one up and interface it?

`train/atari_ff_ppo_gpu.py` is a good place to start for this; replace the env there

If you want something extra back from the environment, look into the collector classes for adding this. Each of the parallel workers will be using a collector which interfaces with the agent through shared buffers?

# Saving + Loading
* Esp. how do you load an agent and then watch it play? E.g. in a notebook. Also, load then evaluate it without watching.

save_itr_params in logger.py
called from minibatch_rl_base.py

Seems like either `logger._snapshot_dir` wasn't set or `_snapshot_mode` was `None`

# Safety Constraints
* With an environment wrapper, this is easy once we figure out custom environments
* And what if I want to just disallow the unsafe actions at the agent level? Need to find the agent probabilities (for on-policy, or Q-values for off) and change those (set some to zero, renormalize) before sampling an action
  * Look in `categorical.py` for changing things for PG agents


# Debugging
* Nice to have an easy way to not use any subprocesses for debugging things
  * Use SerialSampler; maybe note this in a FAQ or Tips section