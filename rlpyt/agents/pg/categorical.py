
import torch

from rlpyt.agents.base import AgentStep, RecurrentAgentMixin
from rlpyt.agents.pg.base import BasePgAgent, AgentInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method


class CategoricalPgAgent(BasePgAgent):

    def __call__(self, observation, prev_action, prev_reward, rgb=None):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        # pi = self.renormalize_safely(observation, pi)
        # TODO - should we zero-out unsafe actions here? This approach of calling
        # is used for training the model, I think, so if we change the distribution,
        # it will affect the loss and the training...
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False):
        super().initialize(env_spaces, share_memory)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, rgb=None):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        pi = self.renormalize_safely(rgb, pi)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def renormalize_safely(self, rgb, pi):
        # if rgb is None or self.checker is None or rgb.ndimension() != 4:
        if rgb.ndimension() != 4:
            return pi

        safe_action_masks = self.checker.get_safe_actions(rgb).to(pi.device)
        # pi[~safe_action_masks] = 0
        pi = pi * safe_action_masks.float()
        pi = pi / pi.sum(-1, keepdim=True)
        return pi

        # TODO - instead of a penalty applied to the reward, you could directly
        # add a term to the loss that penalizes having higher probabilities for
        # unsafe actions. This might be easier to learn than just a reward term
        # which is difficult to associate with the right thing?
        # (for Q networks, you could penalize having higher Q values for unsafe
        # actions, but it's harder there because there isn't a target Q value
        # that you want it to give to unsafe actions (the target probability is 0,
        # but Q values can be negative...))

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value = self.model(*model_inputs)
        return value.to("cpu")


class RecurrentCategoricalPgAgent(RecurrentAgentMixin, BasePgAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        pi, value, next_rnn_state = self.model(*model_inputs)
        dist_info, value = buffer_to((DistInfo(prob=pi), value), device="cpu")
        return dist_info, value, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False):
        super().initialize(env_spaces, share_memory)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
            prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        return value.to("cpu")
