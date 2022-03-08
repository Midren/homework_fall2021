import numpy as np
import torch

from cs285.agents.base_agent import BaseAgent
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicyPG


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(self.agent_params['ac_dim'],
                                 self.agent_params['ob_dim'],
                                 self.agent_params['n_layers'],
                                 self.agent_params['size'],
                                 discrete=self.agent_params['discrete'],
                                 learning_rate=self.agent_params['learning_rate'],
                                 nn_baseline=self.agent_params['nn_baseline'])

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):
        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        q_vals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_vals, terminals)
        return self.actor.update(observations, actions, advantages, q_vals)

    def calculate_q_vals(self, rewards_list):
        """
            Monte Carlo estimation of the Q function.
        """

        if not self.reward_to_go:
            q_estimator_func = self._discounted_return
        else:
            q_estimator_func = self._discounted_cumsum

        return np.concatenate([q_estimator_func(path) for path in rewards_list])

    def estimate_advantage(self, obs, rews_list, q_values, terminals):
        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)

            assert values_unnormalized.ndim == q_values.ndim
            values = values_unnormalized*np.std(q_values)+np.mean(q_values)

            if self.gae_lambda is not None:
                values = np.append(values, [0])

                rews = np.concatenate(rews_list)

                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    if terminals[i]:
                        advantages[i] = rews[i] - values[i]
                    else:
                        advantages[i] = rews[i] + self.gamma*values[i+1] - values[i] + (self.gamma*self.gae_lambda)* advantages[i+1]

                # remove dummy advantage
                advantages = advantages[:-1]
            else:
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = (advantages - np.mean(advantages))/(np.std(advantages)+1e-8)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        return np.ones(len(rewards)) * np.sum(np.array(rewards) * np.geomspace(1, self.gamma**len(rewards), num=len(rewards), endpoint=False))

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        res = [rewards[-1]]
        for reward in rewards[::-1][1:]:
            res.append(res[-1]*self.gamma+reward)

        return np.array(res[::-1])

    def save(self, path):
        torch.save(self.actor, path)
