from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_definition import Simple1DCNN  # 모델 정의를 불러옴

# 데이터 로딩
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test_w_label.csv")

# SubjectID를 정의, 예를 들어 1.0
subject_id = 5.0

# 특정 SubjectID로 train과 val 데이터 필터링
train_data = train_data[train_data["SubjectID"] == subject_id]
test_data = test_data[test_data["SubjectID"] == subject_id]

features = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]
X_train = train_data[features]
X_test = test_data[features]

# 실제 레이블
y_test = test_data["Label"]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_test -= mean
X_test /= std

X_test_tensor = torch.FloatTensor(X_test.values)
test_dataset = TensorDataset(X_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 로드
model = Simple1DCNN(1, 2)
model.load_state_dict(torch.load("./simple_nn_model.pth"))

# 평가
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

# 결과 저장
result = pd.DataFrame({"Predicted_Label": predictions})
result.to_csv("./predictions.csv", index=False)
