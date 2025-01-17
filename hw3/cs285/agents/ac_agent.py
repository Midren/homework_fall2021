import numpy as np
from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
# from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC

from cs285.agents.base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        critic_losses = []
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
            critic_losses.append(critic_loss)

        advantages = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        actor_losses = []
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, advantages)
            actor_losses.append(actor_loss)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_losses
        loss['Actor_Loss'] = actor_losses

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        adv_n = re_n + (self.gamma*self.critic.forward_np(next_ob_no))*np.logical_not(terminal_n) - self.critic.forward_np(ob_no)

        if self.standardize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
