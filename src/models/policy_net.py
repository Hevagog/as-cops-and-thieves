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
        self.len_ch = self.num_observations // 2

        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 13, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def compute(self, inputs, role):
        return (
            self.net(inputs["states"].view(inputs["states"].size(0), 2, self.len_ch)),
            {},
        )

    def act(self, inputs, role=""):
        return super().act(inputs, role)
