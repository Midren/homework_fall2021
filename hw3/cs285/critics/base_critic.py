class BaseCritic:
    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        raise NotImplementedError
