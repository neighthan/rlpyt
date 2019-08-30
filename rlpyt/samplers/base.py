from typing import Any, Dict, Optional

from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save__init__args


class BaseSampler:
    """Class which interfaces with the Runner, in master process only."""

    def __init__(
            self,
            EnvCls,
            env_kwargs: Dict[str, Any],
            batch_T: int,
            batch_B: int,
            max_decorrelation_steps: int=100,
            TrajInfoCls=TrajInfo,
            CollectorCls=None,  # Not auto-populated.
            eval_n_envs: int=0,
            eval_CollectorCls=None,  # Maybe auto-populated.
            eval_env_kwargs : Optional[Dict[str, Any]] = None,
            eval_max_steps: Optional[int] = None,
            eval_max_trajectories: Optional[int] = None,  # Optional earlier cutoff.
            eval_min_envs_reset: int = 1,
            safe: bool = False,
            ):
        """
        :param eval_n_envs: 0 for no eval setup.
        :param eval_env_kwargs: must supply if doing eval.
        :param eval_max_steps: int if using evaluation.
        """
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else
            int(eval_max_trajectories))
        save__init__args(locals())
        self.batch_spec = BatchSpec(batch_T, batch_B)
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr: int):
        raise NotImplementedError

    def evaluate_agent(self, itr: int):
        raise NotImplementedError

    def shutdown(self):
        pass


class BaseCollector:
    """Class that steps through environments, possibly in worker process."""

    def __init__(
            self,
            rank,
            envs,
            samples_np,
            batch_T,
            TrajInfoCls,
            agent=None,  # Present or not, depending on collector class.
            sync=None,
            step_buffer_np=None,
            ):
        save__init__args(locals())

    def start_envs(self):
        """Calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        if getattr(self, "agent", None) is not None:
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """Does not record intermediate data."""

    def __init__(
            self,
            rank,
            envs,
            TrajInfoCls,
            traj_infos_queue,
            max_T,
            agent=None,
            sync=None,
            step_buffer_np=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self):
        raise NotImplementedError
