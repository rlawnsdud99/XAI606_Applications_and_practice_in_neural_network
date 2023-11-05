import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_definition import Simple1DCNN, SimpleMLP
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_and_preprocess_data(subject_id, feature_columns):
    # 데이터 로딩 및 전처리
    # 데이터 로딩
    train_data = pd.read_csv("./train.csv")
    val_data = pd.read_csv("./val.csv")
    test_data = pd.read_csv("./test_w_label.csv")

    # 특정 SubjectID로 train과 val 데이터 필터링
    train_data = train_data[train_data["SubjectID"] == subject_id]
    val_data = val_data[val_data["SubjectID"] == subject_id]
    test_data = test_data[test_data["SubjectID"] == subject_id]
    # Feature 선택 및 데이터 추출
    X_train = train_data[feature_columns]
    y_train = train_data["Label"]
    X_val = val_data[feature_columns]
    y_val = val_data["Label"]
    X_test = test_data[feature_columns]
    y_test = test_data["Label"]

    print(X_train.shape, y_train.shape)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train -= mean
    X_train /= std

    X_val -= mean
    X_val /= std

    X_test -= mean
    X_test /= std

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.LongTensor(y_val.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.LongTensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # ... (이 부분은 원래 코드와 동일)
    return (
        train_loader,
        val_loader,
        test_loader,
        X_test,
        y_test,
        len(feature_columns),
        len(y_train.unique()),
    )


def initialize_model(input_dim, output_dim):
    # 모델 설정
    model = Simple1DCNN(1, output_dim)
    # model = SimpleMLP(1, 32, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0007)
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

    # learning rate 업데이트
    scheduler.step()

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


def visualize_tsne(test_loader, X_test, y_test, model, subject_id, acc_result):
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
    tsne_obj_raw = tsne.fit_transform(X_test.values)

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
    plt.savefig(f"./tsne_result/t-SNE_{subject_id}_{acc_result:.2f}%.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # TensorBoard writer 객체 생성
    writer = SummaryWriter()

    subject_id = 2.0
    epoch = 50
    feature_columns = [
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
        X_test,
        y_test,
        input_dim,
        output_dim,
    ) = load_and_preprocess_data(subject_id, feature_columns)

    model, criterion, optimizer = initialize_model(input_dim, output_dim)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)

    for epoch in range(epoch):  # 50 epochs
        train_model(model, criterion, optimizer, scheduler, train_loader, epoch, writer)
        validate_model(model, criterion, val_loader, epoch, writer)
        scheduler.step()

    # 모델 저장
    torch.save(model.state_dict(), f"./pth/simple_nn_model_{subject_id}.pth")

    # 테스트 데이터에 대한 평가
    acc_result = evaluate_test_model(model, test_loader, y_test, epoch, writer)

    # t-SNE 시각화
    visualize_tsne(test_loader, X_test, y_test, model, subject_id, acc_result)

    # TensorBoard writer 닫기
    writer.close()
