from cs285.infrastructure.utils import *

import numpy as np
import typing as t


class ReplayBuffer:

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs: np.ndarray = None
        self.acs: np.ndarray = None
        self.rews: np.ndarray = None
        self.next_obs: np.ndarray = None
        self.terminals: np.ndarray = None

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.rews = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            if concat_rew:
                self.rews = np.concatenate(
                    [self.rews, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rews += rewards
                else:
                    self.rews.append(rewards)
                self.rews = self.rews[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):
        assert (
                self.obs.shape[0]
                == self.acs.shape[0]
                == self.rews.shape[0]
                == self.next_obs.shape[0]
                == self.terminals.shape[0]
        )

        idx = np.random.permutation(len(self))[:batch_size]


        return (
            self.obs[idx],
            self.acs[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.terminals[idx],
        )

    def sample_recent_data(self, batch_size=1):
        return (
            self.obs[-batch_size:],
            self.acs[-batch_size:],
            self.rews[-batch_size:],
            self.next_obs[-batch_size:],
            self.terminals[-batch_size:],
        )
