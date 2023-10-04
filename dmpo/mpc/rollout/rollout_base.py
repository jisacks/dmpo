class RolloutBase:
    def __init__(self):
        pass

    def cost_fn(self, state, act):
        pass

    def rollout_fn(self, state, act):
        pass

    def current_cost(self, current_state):
        pass

    def update_params(self):
        pass
