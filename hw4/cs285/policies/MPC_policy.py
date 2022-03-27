import numpy as np

from cs285.policies.base_policy import BasePolicy


class MPCPolicy(BasePolicy):
    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, " +
                  f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            return np.random.uniform(self.low, self.high, size=(num_sequences, horizon, self.ac_dim))
        elif self.sample_strategy == 'cem':
            mu = None
            cov = None

            for _ in range(self.cem_iterations):
                if mu is None or cov is None:
                    possible_actions = np.random.uniform(self.low, self.high, size=(num_sequences, horizon, self.ac_dim))
                else:
                    possible_actions = np.random.multivariate_normal(mu, np.diagflat(cov), size=(num_sequences, horizon))

                rewards = np.array([self.evaluate_candidate_sequences(action_seq, obs) for action_seq in possible_actions])
                top_idx = np.argpartition(rewards, -self.cem_num_elites)[-self.cem_num_elites:]
                a_elites = rewards[top_idx]

                if mu is None or cov is None:
                    mu = np.mean(a_elites, axis=0)
                    cov = np.var(a_elites, axis=0)
                else:
                    mu = self.cem_alpha * np.mean(a_elites, axis=0) + (1 - self.cem_alpha) * mu
                    cov = self.cem_alpha * np.var(a_elites, axis=0) + (1 - self.cem_alpha) * cov

                assert mu.shape == (horizon, self.ac_dim)
                assert cov.shape == (horizon, self.ac_dim)

            cem_action = mu

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        return sum(self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
                   for model in self.dyn_models) / len(self.dyn_models)

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)]
            action_to_take = best_action_sequence[0]
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        sum_of_rewards = [0 for _ in range(self.N)]

        curr_ob = np.ones((self.N, len(obs)))*obs
        for step_idx in range(self.horizon):
            curr_ob = model.get_prediction(curr_ob, candidate_action_sequences[:, step_idx], self.data_statistics)
            sum_of_rewards += self.env.get_reward(curr_ob, candidate_action_sequences[:, step_idx])[0]
        return np.array(sum_of_rewards)
