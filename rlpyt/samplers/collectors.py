from typing import Optional
import numpy as np

from rlpyt.samplers.base import BaseCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class DecorrelatingStartCollector(BaseCollector):

    def start_envs(self, max_decorrelation_steps: int=0) -> AgentInputs:
        """Calls reset() on every env and returns agent_inputs buffer."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        rgbs = list()
        for env in self.envs:
            obs, rgb = env.reset(rgb=True)
            observations.append(obs)
            rgbs.append(rgb)
        observation = buffer_from_example(observations[0], len(self.envs))
        rgb = buffer_from_example(rgbs[0], len(self.envs))
        for b, obs in enumerate(observations):
            observation[b] = obs  # numpy array or namedarraytuple
            rgb[b] = rgbs[b]
        prev_action = self.envs[0].action_space.sample(len(self.envs), null=True)
        prev_reward = np.zeros(len(self.envs), dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                f"{max_decorrelation_steps}")
        if max_decorrelation_steps == 0:
            return AgentInputs(observation, prev_action, prev_reward, rgb), traj_infos
        for b, env in enumerate(self.envs):
            n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
            env_actions = env.action_space.sample(n_steps)
            for a in env_actions:
                a = np.array(a)
                (o, r, d, info), rgb_ = env.step(a, ignore_safe_act_method=True, rgb=True)
                traj_infos[b].step(o, a.flat[0], r, d, None, info)
                if getattr(info, "traj_done", d):
                    o = env.reset()
                    traj_infos[b] = self.TrajInfoCls()
                if d:
                    a = env.action_space.sample(null=True)
                    r = 0
            observation[b] = o
            prev_action[b] = a
            prev_reward[b] = r
            rgb[b] = rgb_
        return AgentInputs(observation, prev_action, prev_reward, rgb), traj_infos


class SerialEvalCollector:
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T: int,
            max_trajectories: Optional[int] = None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr: int) -> list:
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        action = buffer_from_example(self.envs[0].action_space.sample(null=True),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b], agent_info, info_idx=b)
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                break
        return completed_traj_infos
