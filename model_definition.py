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
    def __init__(self, num_channel, num_classes, num_feature):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channel, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(
            num_feature * 128, 128
        )  # You might need to adjust the dimension here
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)  # 드롭아웃 추가

    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.elu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Larger1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Larger1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)  # 추가된 레이어
        self.fc1 = nn.Linear(256, 512)  # 차원 조정 필요
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  # 드롭아웃 추가

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))  # 추가된 레이어
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Improved1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Improved1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.adapool = nn.AdaptiveAvgPool1d(5)
        self.fc1 = nn.Linear(256 * 5, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        # Loop over all modules and apply the He initialization to convolutional and linear layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Define a simple Multi-Layer Perceptron (MLP) model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x
