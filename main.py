import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from model_definition import AttentionModel, Simple1DCNN

import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# TensorBoard writer 객체 생성
writer = SummaryWriter()

# 데이터 로딩
train_data = pd.read_csv("./train.csv")
val_data = pd.read_csv("./val.csv")

# SubjectID를 정의, 예를 들어 1.0
subject_id = 5.0

# 특정 SubjectID로 train과 val 데이터 필터링
train_data = train_data[train_data["SubjectID"] == subject_id]
val_data = val_data[val_data["SubjectID"] == subject_id]

# Feature 선택 및 데이터 추출
features = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]
X_train = train_data[features]
y_train = train_data["Label"]
X_val = val_data[features]
y_val = val_data["Label"]

print(X_train.shape, y_train.shape)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train -= mean
X_train /= std

X_val -= mean
X_val /= std


X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.LongTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val.values)
y_val_tensor = torch.LongTensor(y_val.values)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# 모델 정의
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


# 모델 설정
input_dim = len(features)
output_dim = len(y_train.unique())

# model = AttentionModel(input_dim, output_dim, d_model, num_heads)
model = Simple1DCNN(1, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)
epoch = 50
scheduler = CosineAnnealingLR(optimizer, T_max=epoch)  # T_max는 전체 epoch 수

# 학습
for epoch in range(epoch):  # 500 epochs
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0  # 여기 추가
    for i, batch in enumerate(train_loader):
        x_batch, y_batch = batch
        x_batch = x_batch.unsqueeze(1)  # Adding a channel dimension
        optimizer.zero_grad()
        output = model(x_batch)
        _, predicted_train = torch.max(output.data, 1)  # 여기 추가
        total_train += y_batch.size(0)  # 여기 추가
        correct_train += (predicted_train == y_batch).sum().item()  # 여기 추가
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_accuracy = 100 * correct_train / total_train  # 여기 추가
    avg_train_loss = running_loss / len(train_loader)

    # TensorBoard에 train loss와 train accuracy 로깅
    writer.add_scalar("Training loss", avg_train_loss, epoch)
    writer.add_scalar("Training Accuracy", train_accuracy, epoch)  # 여기 추가

    # 검증
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0  #
    with torch.no_grad():
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.unsqueeze(1)
            val_output = model(x_val_batch)
            val_loss = criterion(val_output, y_val_batch)  #
            running_val_loss += val_loss.item()  #
            _, predicted = torch.max(val_output.data, 1)
            total += y_val_batch.size(0)
            correct += (predicted == y_val_batch).sum().item()
    avg_val_loss = running_val_loss / len(val_loader)  #
    val_accuracy = 100 * correct / total

    # TensorBoard에 로깅
    writer.add_scalar("Validation loss", avg_val_loss, epoch)  #
    writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

    print(
        f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {avg_val_loss:.2f}"
    )

    # learning rate 업데이트
    scheduler.step()

# 모델 저장
torch.save(model.state_dict(), "./simple_nn_model.pth")

# TensorBoard writer 닫기
writer.close()
# tensorboard --logdir=runs
