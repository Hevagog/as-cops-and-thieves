import torch
import torch.nn as nn
from skrl.models.torch import Model, CategoricalMixin


class LSTMPolicy(CategoricalMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        unnormalized_log_prob=True,
        num_envs=1,
        num_layers=1,
        hidden_size=128,
        sequence_length=16,  # Longer for spatial memory
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.len_ch = self.num_observations // 2

        # Feature extraction (similar to your current approach)
        self.features_extractor = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 13, 256),
            nn.Tanh(),
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=256,  # Output from features_extractor
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states
                    (self.num_layers, self.num_envs, self.hidden_size),  # cell states
                ],
            }
        }

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)

        # Handle missing RNN states on first run
        if "rnn" in inputs:
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
        else:
            # Initialize RNN states for first run
            batch_size = (
                states.size(0) // self.sequence_length
                if states.size(0) >= self.sequence_length
                else 1
            )
            hidden_states = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )
            cell_states = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )

        # Extract features from observations
        batch_size = states.size(0)
        features = self.features_extractor(states.view(batch_size, 2, self.len_ch))

        # Prepare for LSTM
        # Handle case where batch_size < sequence_length (first few steps)
        if batch_size < self.sequence_length:
            # Pad the sequence with zeros
            pad_size = self.sequence_length - batch_size
            padding = torch.zeros(
                pad_size,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            features_padded = torch.cat([features, padding], dim=0)
            rnn_input = features_padded.unsqueeze(0)  # Add batch dimension
        else:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])

        # Handle RNN states dimensions
        if hidden_states.dim() == 4:  # If states have sequence dimension
            sequence_index = 1 if role == "target_policy" else 0
            hidden_states = hidden_states[:, :, sequence_index, :].contiguous()
            cell_states = cell_states[:, :, sequence_index, :].contiguous()

        # Handle terminations
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = (
                terminated.view(-1, self.sequence_length)
                if terminated.numel() >= self.sequence_length
                else terminated.view(1, -1)
            )

            # Ensure terminated tensor has correct sequence length
            if terminated.size(1) < self.sequence_length:
                pad_size = self.sequence_length - terminated.size(1)
                term_padding = torch.zeros(
                    terminated.size(0),
                    pad_size,
                    device=terminated.device,
                    dtype=terminated.dtype,
                )
                terminated = torch.cat([terminated, term_padding], dim=1)

            indexes = (
                [0]
                + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                + [self.sequence_length]
            )

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, (hidden_states, cell_states) = self.lstm(
                    rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                )
                if i1 - 1 < terminated.size(1):
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_states = (hidden_states, cell_states)
            rnn_output = torch.cat(rnn_outputs, dim=1)
        else:
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # Flatten and get policy logits
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        # Handle case where we have fewer outputs than expected batch size
        if rnn_output.size(0) > batch_size:
            rnn_output = rnn_output[:batch_size]  # Trim padding

        policy_logits = self.policy_head(rnn_output)

        return policy_logits, {"rnn": [rnn_states[0], rnn_states[1]]}

    def act(self, inputs, role=""):
        return super().act(inputs, role)
