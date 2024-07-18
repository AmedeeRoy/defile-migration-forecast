import numpy as np
import torch
import torch.nn as nn


# from https://medium.com/@mkaanaslan99/time-series-forecasting-with-a-basic-transformer-model-in-pytorch-650f116a1018
class transformer_block(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob):
        super(transformer_block, self).__init__()

        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.LeakyReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)
        return x


# from https://stackoverflow.com/questions/68477306/positional-encoding-for-time-series-based-data-for-transformer-dnn-models
class PositionalEncodingLayer(nn.Module):
    def __init__(self, dim):
        super(PositionalEncodingLayer, self).__init__()
        self.dim = dim

    def get_angles(self, positions, indexes):
        dim_tensor = torch.FloatTensor([[self.dim]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / dim_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """:param Tensor[batch_size, seq_len] input_sequences :return Tensor[batch_size, seq_len,
        dim] position_encoding."""
        positions = (
            torch.arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device)
        )  # [seq_len, 1]
        indexes = torch.arange(self.dim).unsqueeze(0).to(input_sequences.device)  # [1, dim]
        angles = self.get_angles(positions, indexes)  # [seq_len, dim]
        angles[:, 0::2] = torch.sin(angles[:, 0::2])  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2])  # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(
            input_sequences.size(0), 1, 1
        )  # [batch_size, seq_len, dim]
        return position_encoding


class Transformer(nn.Module):
    def __init__(
        self,
        nb_input_features,
        embed_size_hourly,
        num_heads_hourly,
        num_blocks_hourly,
        embed_size_daily,
        num_heads_daily,
        num_blocks_daily,
        drop_prob,
    ):
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEncodingLayer(dim=nb_input_features)

        # Hourly Network --------------------------

        self.cnn_embedding_hourly = nn.Sequential(
            nn.Conv1d(
                nb_input_features,
                embed_size_hourly,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1,
            ),
            nn.BatchNorm1d(num_features=embed_size_hourly),
            nn.LeakyReLU(),
        )

        self.blocks_hourly = nn.ModuleList(
            [
                transformer_block(embed_size_hourly, num_heads_hourly, drop_prob)
                for n in range(num_blocks_hourly)
            ]
        )

        self.cnn_output_hourly = nn.Sequential(
            nn.Conv1d(embed_size_hourly, 1, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.Sigmoid(),
        )

        # Daily Network --------------------------

        self.cnn_embedding_daily = nn.Sequential(
            nn.Conv1d(
                nb_input_features, embed_size_daily, kernel_size=5, stride=1, padding=2, dilation=1
            ),
            nn.BatchNorm1d(num_features=embed_size_daily),
            nn.LeakyReLU(),
        )

        self.blocks_daily = nn.ModuleList(
            [
                transformer_block(embed_size_daily, num_heads_daily, drop_prob)
                for n in range(num_blocks_daily)
            ]
        )

        self.last_layer_daily = nn.Sequential(
            nn.Linear(embed_size_daily, 1),
            nn.Sigmoid(),
        )

    def forward(self, yr, doy, era5_hourly, era5_daily):
        # Hourly Transformer
        doy_ = doy.repeat(1, 24).unsqueeze(1)
        yr_ = yr.repeat(1, 24).unsqueeze(1)
        x_h = torch.cat([era5_hourly, doy_, yr_], 1)
        x_h = x_h + self.positional_encoding(x_h.transpose(1, 2)).transpose(1, 2)

        out_h = self.cnn_embedding_hourly(x_h)
        out_h = out_h.transpose(1, 2)
        for block in self.blocks_hourly:
            out_h = block(out_h)
        out_h = out_h.transpose(1, 2)
        out_h = self.cnn_output_hourly(out_h)

        # Daily Transformer
        doy_ = doy.repeat(1, 7).unsqueeze(1)
        yr_ = yr.repeat(1, 7).unsqueeze(1)
        x_d = torch.cat([era5_daily, doy_, yr_], 1)
        x_d = x_d + self.positional_encoding(x_d.transpose(1, 2)).transpose(1, 2)

        out_d = self.cnn_embedding_daily(x_d)
        out_d = out_d.transpose(1, 2)
        for block in self.blocks_daily:
            out_d = block(out_d)
        out_d = torch.mean(out_d, dim=1)
        out_d = self.last_layer_daily(out_d).unsqueeze(1)

        out = 5 * out_h * out_d

        # Force count to be zero between 0-? and ?-24 hr
        pred_mask = np.array([1 for i in range(24)])
        pred_mask[:6] = 0
        pred_mask[21:] = 0
        pred_mask = torch.FloatTensor(pred_mask).repeat(out.shape[0], 1).unsqueeze(1)
        out = out * pred_mask

        return out
