import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_definition import Simple1DCNN, Improved1DCNN, Larger1DCNN
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def augment_data(X, y, num_augmented=3, noise_level=0.02):
    # X와 y는 numpy 배열이어야 함
    augmented_X = X.copy()
    augmented_y = y.copy()
    for _ in range(num_augmented):
        noise = np.random.normal(0, noise_level, X.shape)
        new_X = X + noise
        augmented_X = np.vstack((augmented_X, new_X))
        augmented_y = np.concatenate((augmented_y, y))  # y는 늘려준 X와 같은 수만큼 반복
    return augmented_X, augmented_y


def load_and_preprocess_data(feature_columns, subject_id):
    # 데이터 로딩 및 전처리
    # 데이터 로딩
    train_data = pd.read_csv("./train.csv")
    val_data = pd.read_csv("./val.csv")
    test_data = pd.read_csv("./test_w_label.csv")

    # 특정 SubjectID로 train과 val 데이터 필터링
    train_data = train_data[train_data["SubjectID"] == subject_id]
    val_data = val_data[val_data["SubjectID"] == subject_id]
    test_data = test_data[test_data["SubjectID"] == subject_id]

    X_train = train_data[feature_columns].values
    y_train = train_data["Label"].values
    X_val = val_data[feature_columns].values
    y_val = val_data["Label"].values
    X_test = test_data[feature_columns].values
    y_test = test_data["Label"].values

    X_train, y_train = augment_data(X_train, y_train)
    print(X_train.shape, y_train.shape)
    norm = StandardScaler()
    X_train_scaled = norm.fit_transform(X_train)
    X_val_scaled = norm.transform(X_val)
    X_test_scaled = norm.transform(X_test)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # ... (이 부분은 원래 코드와 동일)
    return (
        train_loader,
        val_loader,
        test_loader,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        len(feature_columns),
        len(np.unique(y_train)),
    )


def data_statistics(X_train_scaled):
    # 상관 관계 계산
    correlation_matrix = pd.DataFrame(X_train_scaled, columns=feature_columns).corr()

    # 상관 관계 히트맵 그리기
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    # 각 feature의 분포를 히스토그램으로 그리기
    pd.DataFrame(X_train_scaled, columns=feature_columns).hist(
        bins=15, figsize=(15, 10)
    )
    plt.suptitle("Feature Distribution Histograms")
    plt.show()


def initialize_model(input_dim, output_dim):
    # 모델 설정
    model = Simple1DCNN(1, output_dim, input_dim)
    # model = Improved1DCNN(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.0007)
    return model, criterion, optimizer


def train_model(model, criterion, optimizer, scheduler, train_loader, epoch, writer):
    # 학습
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
    print(
        f"Epoch {epoch+1}, Training Accuracy: {train_accuracy:.2f}%, Tranining Loss: {avg_train_loss:.2f}"
    )
    # ... (이 부분은 원래 코드와 동일)


def validate_model(model, criterion, val_loader, epoch, writer):
    # 검증
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
    return val_accuracy
    # ... (이 부분은 원래 코드와 동일)


def evaluate_test_model(model, test_loader, y_test, epoch, writer):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_test_batch in test_loader:
            x_test_batch = x_test_batch[0].unsqueeze(1)
            output = model(x_test_batch)
            _, predicted = torch.max(output.data, 1)
            predictions.extend(predicted.tolist())

    # 정확도 산출
    accuracy = accuracy_score(y_test, predictions) * 100  # 실제 레이블과 비교
    print(f"Test Accuracy: {accuracy:.2f}%")

    # TensorBoard에 로깅
    writer.add_scalar("Test Accuracy", accuracy, epoch)
    return accuracy


def visualize_tsne(test_loader, X_test, y_test, model, acc_result):
    feature_outputs_list = []

    # 모델을 통과시켜 특징 추출
    with torch.no_grad():
        for x_test_batch in test_loader:
            x_test_batch = x_test_batch[0].unsqueeze(1)
            feature_outputs = model(x_test_batch)
            feature_outputs_list.append(feature_outputs)

    feature_outputs_all = torch.cat(feature_outputs_list, axis=0)

    # t-SNE 객체 생성
    tsne = TSNE(n_components=2, random_state=0)

    # 원본 데이터에 t-SNE 적용
    tsne_obj_raw = tsne.fit_transform(X_test)

    # 모델 출력에 t-SNE 적용
    tsne_obj_model = tsne.fit_transform(feature_outputs_all.numpy())

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 원본 데이터 t-SNE
    scatter = axes[0].scatter(
        tsne_obj_raw[:, 0], tsne_obj_raw[:, 1], c=y_test, cmap="viridis"
    )
    axes[0].set_title("t-SNE on Raw Data")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].legend(
        handles=scatter.legend_elements()[0], labels=set(y_test), title="Classes"
    )

    # 모델 출력 t-SNE
    scatter = axes[1].scatter(
        tsne_obj_model[:, 0], tsne_obj_model[:, 1], c=y_test, cmap="viridis"
    )
    axes[1].set_title(f"t-SNE on Model Output: Acc={acc_result:.2f}")
    axes[1].set_xlabel("Dimension 1")
    axes[1].set_ylabel("Dimension 2")
    axes[1].legend(
        handles=scatter.legend_elements()[0], labels=set(y_test), title="Classes"
    )

    plt.tight_layout()
    # 파일로 저장
    plt.savefig(f"./tsne_result/t-SNE_{acc_result:.2f}%.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # TensorBoard writer 객체 생성
    writer = SummaryWriter()
    subject_id = 4.0
    epoch = 100
    feature_columns = [
        "Raw",
        "Delta",
        "Theta",
        "Alpha1",
        "Alpha2",
        "Beta1",
        "Beta2",
        "Gamma1",
        "Gamma2",
    ]
    (
        train_loader,
        val_loader,
        test_loader,
        X_train,
        y_train,
        X_test,
        y_test,
        input_dim,
        output_dim,
    ) = load_and_preprocess_data(feature_columns, subject_id)
    # data_statistics(X_train)
    print(input_dim, output_dim)
    model, criterion, optimizer = initialize_model(input_dim, output_dim)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)
    print(model)
    best_val_accuracy = 0.0  # 최고 정확도를 기록할 변수 초기화
    for epoch in range(epoch):  # 50 epochs
        train_model(model, criterion, optimizer, scheduler, train_loader, epoch, writer)
        val_accuracy = validate_model(model, criterion, val_loader, epoch, writer)
        scheduler.step()
        # Log current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)
        # 만약 현재 epoch에서의 val_accuracy가 이전의 best_val_accuracy보다 높다면 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy  # 최고 정확도 업데이트
            # 모델 저장
            torch.save(
                model.state_dict(), f"./pth/best_model_{best_val_accuracy:.4f}.pth"
            )
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
    # 모델 저장
    torch.save(model.state_dict(), f"./pth/simple_nn_model.pth")

    # 테스트 데이터에 대한 평가
    acc_result = evaluate_test_model(model, test_loader, y_test, epoch, writer)

    # t-SNE 시각화
    visualize_tsne(test_loader, X_test, y_test, model, acc_result)

    # TensorBoard writer 닫기
    writer.close()
# tensorboard --logdir=./runs/


# # 히스토그램 그리기
# train_data[feature_columns].hist(bins=50, figsize=(20, 15))
# plt.suptitle("Train Data Feature Distribution")
# plt.show()

# val_data[feature_columns].hist(bins=50, figsize=(20, 15))
# plt.suptitle("Validation Data Feature Distribution")
# plt.show()

# # 박스플롯 그리기
# plt.figure(figsize=(12, 6))
# sns.boxplot(data=train_data[feature_columns])
# plt.title("Train Data Boxplot")
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.boxplot(data=val_data[feature_columns])
# plt.title("Validation Data Boxplot")
# plt.show()

# # 상관 계수 히트맵
# plt.figure(figsize=(10, 8))
# sns.heatmap(train_data[feature_columns].corr(), annot=True, fmt=".2f")
# plt.title("Train Data Correlation Matrix")
# plt.show()

# plt.figure(figsize=(10, 8))
# sns.heatmap(val_data[feature_columns].corr(), annot=True, fmt=".2f")
# plt.title("Validation Data Correlation Matrix")
# plt.show()
# # Feature 선택 및 데이터 추출
