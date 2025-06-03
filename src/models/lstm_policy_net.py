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
        rnn_from_input = "rnn" in inputs

        # Determine initial hidden_states and cell_states
        if rnn_from_input:
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
        else:
            # Initialize RNN states for first run
            # Calculate effective batch size for LSTM input, matching how rnn_input.size(0) will be calculated
            current_batch_size = states.size(0)
            if current_batch_size < self.sequence_length:
                expected_lstm_batch_dim = 1
            else:
                # Equivalent to ceil(current_batch_size / self.sequence_length)
                expected_lstm_batch_dim = (
                    current_batch_size + self.sequence_length - 1
                ) // self.sequence_length

            hidden_states = torch.zeros(
                self.num_layers,
                expected_lstm_batch_dim,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )
            cell_states = torch.zeros(
                self.num_layers,
                expected_lstm_batch_dim,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )

        # Extract features from observations
        batch_size = states.size(0)  # This is the original inputs["states"].size(0)
        features = self.features_extractor(states.view(batch_size, 2, self.len_ch))

        # Prepare features for LSTM input (rnn_input)
        # rnn_input must have shape (num_sequences, sequence_length, feature_dim)
        if batch_size < self.sequence_length:
            # If batch_size is less than sequence_length, pad features to sequence_length.
            # The LSTM will see this as a single sequence.
            pad_size = self.sequence_length - batch_size
            padding = torch.zeros(
                pad_size,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            # features_for_lstm shape: (sequence_length, feature_dim)
            features_for_lstm = torch.cat([features, padding], dim=0)
            # rnn_input shape: (1, sequence_length, feature_dim)
            rnn_input = features_for_lstm.unsqueeze(0)
        else:  # batch_size >= self.sequence_length
            # If batch_size is greater or equal, features might need padding
            # to make its first dimension a multiple of sequence_length.
            features_for_lstm = features  # Initialize with original features
            if batch_size % self.sequence_length != 0:
                # Calculate padding needed to make features.size(0) a multiple of sequence_length
                num_sequences_to_form = (
                    batch_size + self.sequence_length - 1
                ) // self.sequence_length
                required_total_rows = num_sequences_to_form * self.sequence_length
                padding_rows_needed = required_total_rows - batch_size

                # padding_rows_needed will be > 0 here because batch_size % self.sequence_length != 0
                padding = torch.zeros(
                    padding_rows_needed,
                    features.shape[-1],
                    device=features.device,
                    dtype=features.dtype,
                )
                # features_for_lstm shape: (required_total_rows, feature_dim)
                features_for_lstm = torch.cat([features, padding], dim=0)
            # else: batch_size is already a multiple of sequence_length. No padding needed.
            # features_for_lstm remains as original features.

            # Reshape features_for_lstm into sequences for LSTM
            # features_for_lstm.size(0) is now guaranteed to be a multiple of self.sequence_length
            # rnn_input shape: (N, sequence_length, feature_dim), where N = features_for_lstm.size(0) / sequence_length
            rnn_input = features_for_lstm.view(
                -1, self.sequence_length, features_for_lstm.shape[-1]
            )

        actual_lstm_input_batch_dim = rnn_input.size(0)

        # Adjust hidden_states if they came from input and mismatch rnn_input's batch dim
        if rnn_from_input:
            if hidden_states.size(1) != actual_lstm_input_batch_dim:
                # This warning can be helpful for debugging configuration issues
                # print(f"Warning: LSTMPolicy.compute: Mismatch between hidden_state batch_dim {hidden_states.size(1)} "
                #       f"and rnn_input batch_dim {actual_lstm_input_batch_dim}. Adjusting hidden_state.")

                h_current_bs = hidden_states.size(1)
                if h_current_bs < actual_lstm_input_batch_dim:
                    # Pad hidden_states
                    h_padding_needed = actual_lstm_input_batch_dim - h_current_bs
                    h_pad_tensor = torch.zeros(
                        hidden_states.size(0),
                        h_padding_needed,
                        hidden_states.size(2),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    hidden_states = torch.cat((hidden_states, h_pad_tensor), dim=1)

                    c_pad_tensor = torch.zeros(
                        cell_states.size(0),
                        h_padding_needed,
                        cell_states.size(2),
                        device=cell_states.device,
                        dtype=cell_states.dtype,
                    )
                    cell_states = torch.cat((cell_states, c_pad_tensor), dim=1)
                else:  # h_current_bs > actual_lstm_input_batch_dim
                    # Truncate hidden_states
                    hidden_states = hidden_states[:, :actual_lstm_input_batch_dim, :]
                    cell_states = cell_states[:, :actual_lstm_input_batch_dim, :]

        # Handle RNN states dimensions (e.g. if 4D from skrl)
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
                    hidden_states[:, (terminated[:, i1 - 1]).bool(), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]).bool(), :] = 0
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
