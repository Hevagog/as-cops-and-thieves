import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin


class Value(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        agent_count,
        clip_actions=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return (
            self.net(inputs["states"]),
            {},
        )
