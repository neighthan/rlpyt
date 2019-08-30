
import torch

from rlpyt.agents.base import AgentStep, RecurrentAgentMixin
from rlpyt.agents.pg.base import BasePgAgent, AgentInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method


class CategoricalPgAgent(BasePgAgent):

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        # if self.checker:
        #     pi = self.renormalize_safely(observation, pi)
        # TODO - should we zero-out unsafe actions here? This approach of calling
        # is used for training the model, I think, so if we change the distribution,
        # it will affect the loss and the training...
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False):
        super().initialize(env_spaces, share_memory)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward, safe: bool = True):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        # TODO - should I actually renormalize pi or just sample `action` from a
        # renormalized version and leave pi as-is?
        if self.checker and safe:
            pi, constraint_used = self.renormalize_safely(observation, pi)
        else:
            constraint_used = torch.zeros_like(value)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value, constraint_used=constraint_used)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def renormalize_safely(self, observation, pi):
        squeeze_pi = False
        if observation.ndimension() != 4:
            squeeze_pi = True
            observation = observation.unsqueeze(0)
            pi = pi.unsqueeze(0)

        safe_action_masks = self.checker.get_safe_actions(observation).to(pi.device)
        # TODO - do .any and .where here work as desired? Haven't tested
        safe_action_exists = safe_action_masks.any(-1, keepdim=True).expand(-1, pi.shape[1])
        # pi[~safe_action_masks] = 0
        pi = torch.where(safe_action_exists, pi * safe_action_masks.float(), pi)
        pi_sum = pi.sum(-1, keepdim=True)
        constraint_used = 1 - pi_sum.squeeze()
        pi = pi / pi_sum

        if squeeze_pi:
            pi = pi.squeeze(0)
        return pi, constraint_used

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
