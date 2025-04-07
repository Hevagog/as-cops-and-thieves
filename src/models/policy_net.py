from skrl.models.torch import Model, CategoricalMixin
import torch.nn as nn


class Policy(CategoricalMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        unnormalized_log_prob=True,
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

    def act(self, inputs, role=""):
        return super().act(inputs, role)
