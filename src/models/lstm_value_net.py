import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin


class LSTMValue(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,  # This is the Dict space for shared observations for one agent
        action_space,
        device,
        agent_count,
        clip_actions=False,
        num_envs=1,
        sequence_length=16,
        hidden_size=128,
        num_layers=2,
        # Parameters for the new CNN feature extractor
        cnn_input_channels=4,  # own_obj_types, own_distances, object_type_shared, distance_shared
        cnn_output_features=256,  # Desired output size from CNN block, to be LSTM input size
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.agent_count = agent_count

        # Determine the length of each channel (e.g., num_rays)
        # Assuming all 4 observation components have the same shape (num_rays,)
        # Keys in observation_space are typically sorted alphabetically by skrl for flattening
        # 'distance_shared', 'object_type_shared', 'own_distances', 'own_obj_types', 'team_positions'
        self.cnn_channel_length = 90  # Staticly added for now
        self.cnn_input_channels = cnn_input_channels

        # CNN Feature extractor (inspired by LSTMPolicy)
        # Calculate L_out for the Linear layer based on Conv1D parameters and cnn_channel_length
        # L_out = floor((L_in - kernel_size + 2*padding) / stride + 1)
        # Conv1: kernel=5, stride=2, padding=0
        l_out_conv1 = (self.cnn_channel_length - 5) // 2 + 1
        # Conv2: kernel=5, stride=3, padding=0
        l_out_conv2 = (l_out_conv1 - 5) // 3 + 1

        self.features_extractor = nn.Sequential(
            nn.Conv1d(self.cnn_input_channels, 64, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * l_out_conv2, cnn_output_features),
            nn.Tanh(),
        )

        lstm_input_size = cnn_output_features

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,  # Input is now from the CNN features_extractor
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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
        states = inputs["states"]  # Flattened shared observation for the agent
        terminated = inputs.get("terminated", None)
        batch_size = states.size(0)

        # Calculate the batch dimension that rnn_input will have (number of sequences)
        if batch_size < self.sequence_length:
            expected_rnn_input_batch_dim = 1
        else:
            # Equivalent to ceil(batch_size / self.sequence_length)
            expected_rnn_input_batch_dim = (
                batch_size + self.sequence_length - 1
            ) // self.sequence_length

        rnn_states_from_input = "rnn" in inputs
        if rnn_states_from_input:
            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]
        else:
            # Initialize RNN states for first run using the correctly calculated batch dimension
            hidden_states = torch.zeros(
                self.num_layers,
                expected_rnn_input_batch_dim,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )
            cell_states = torch.zeros(
                self.num_layers,
                expected_rnn_input_batch_dim,
                self.hidden_size,
                device=self.device,
                dtype=states.dtype,
            )

        L = self.cnn_channel_length

        own_obj_types_data = states[:, 3 * L : 4 * L]
        own_distances_data = states[:, 2 * L : 3 * L]
        object_type_shared_data = states[:, 1 * L : 2 * L]
        distance_shared_data = states[:, 0 * L : 1 * L]

        cnn_input_tensor = torch.stack(
            [
                own_obj_types_data,
                own_distances_data,
                object_type_shared_data,
                distance_shared_data,
            ],
            dim=1,
        ).view(batch_size, self.cnn_input_channels, self.cnn_channel_length)

        features = self.features_extractor(
            cnn_input_tensor
        )  # (batch_size, cnn_output_features)

        # Prepare for LSTM
        # Handle case where batch_size < sequence_length (first few steps during rollout)
        if batch_size < self.sequence_length:
            pad_size = self.sequence_length - batch_size
            padding = torch.zeros(
                pad_size,
                features.shape[-1],
                device=features.device,
                dtype=features.dtype,
            )
            features_padded = torch.cat([features, padding], dim=0)
            rnn_input = features_padded.unsqueeze(0)  # Add batch dimension for LSTM
        else:  # batch_size >= self.sequence_length
            features_for_lstm = features
            if batch_size % self.sequence_length != 0:
                num_sequences_to_form = (
                    batch_size + self.sequence_length - 1
                ) // self.sequence_length
                required_total_rows = num_sequences_to_form * self.sequence_length
                padding_rows_needed = required_total_rows - batch_size
                padding = torch.zeros(
                    padding_rows_needed,
                    features.shape[-1],
                    device=features.device,
                    dtype=features.dtype,
                )
                features_for_lstm = torch.cat([features, padding], dim=0)

            rnn_input = features_for_lstm.view(
                -1, self.sequence_length, features_for_lstm.shape[-1]
            )

        # At this point, rnn_input.size(0) should be equal to expected_rnn_input_batch_dim.
        # Adjust hidden_states if they came from input and mismatch rnn_input's batch dim
        if rnn_states_from_input:
            if hidden_states.size(1) != expected_rnn_input_batch_dim:
                h_current_bs = hidden_states.size(1)
                if h_current_bs < expected_rnn_input_batch_dim:
                    # Pad hidden_states and cell_states
                    padding_needed = expected_rnn_input_batch_dim - h_current_bs
                    h_pad_tensor = torch.zeros(
                        hidden_states.size(0),
                        padding_needed,
                        hidden_states.size(2),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                    hidden_states = torch.cat((hidden_states, h_pad_tensor), dim=1)

                    c_pad_tensor = torch.zeros(
                        cell_states.size(0),
                        padding_needed,
                        cell_states.size(2),
                        device=cell_states.device,
                        dtype=cell_states.dtype,
                    )
                    cell_states = torch.cat((cell_states, c_pad_tensor), dim=1)
                else:  # h_current_bs > expected_rnn_input_batch_dim
                    # Truncate hidden_states and cell_states
                    hidden_states = hidden_states[:, :expected_rnn_input_batch_dim, :]
                    cell_states = cell_states[:, :expected_rnn_input_batch_dim, :]

        # Handle RNN states dimensions
        if hidden_states.dim() == 4:
            sequence_index = 1 if role == "target_critic" else 0
            hidden_states = hidden_states[:, :, sequence_index, :].contiguous()
            cell_states = cell_states[:, :, sequence_index, :].contiguous()

        # Handle terminations (similar logic as policy network)
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = (
                terminated.view(-1, self.sequence_length)
                if terminated.numel() >= self.sequence_length
                else terminated.view(1, -1)
            )

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
                # Pass the correct slice of rnn_input to LSTM
                rnn_output_segment, (hidden_states, cell_states) = self.lstm(
                    rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                )
                if i1 - 1 < terminated.size(1):
                    # Use .bool() for indexing if terminated can be float
                    hidden_states[:, (terminated[:, i1 - 1]).bool(), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]).bool(), :] = 0
                rnn_outputs.append(rnn_output_segment)

            rnn_states = (hidden_states, cell_states)
            rnn_output = torch.cat(rnn_outputs, dim=1)
        else:
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # Flatten and get value
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        if rnn_output.size(0) > batch_size:
            rnn_output = rnn_output[:batch_size]

        value = self.value_head(rnn_output)

        return value, {"rnn": [rnn_states[0], rnn_states[1]]}
