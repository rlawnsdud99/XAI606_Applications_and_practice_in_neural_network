import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(self.head_dim, self.head_dim)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split embedding into self.num_heads different pieces
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)
        key = key.reshape(N, key_len, self.num_heads, self.head_dim)
        value = value.reshape(N, value_len, self.num_heads, self.head_dim)

        scores = torch.einsum("nqhd,nkhd->nhqk", [query, key])
        attention = torch.nn.functional.softmax(scores, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, value]).reshape(
            N, query_len, self.d_model
        )

        out = self.fc_out(out)
        return out


# Simple model with Attention Mechanism
class AttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads):
        super(AttentionModel, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.attention(x, x, x)
        x = self.fc(x)
        return x


class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.5)  # Dropout layer after conv2
        self.fc1 = nn.Linear(64, 128)  # You might need to adjust the dimension here
        self.dropout2 = nn.Dropout(
            0.5
        )  # Dropout layer before the final fully connected layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)  # Dropout after conv2
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # Dropout before the final layer
        x = self.fc2(x)
        return x
