from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import pandas as pd
from model_definition import Simple1DCNN

# 모델 불러오기
model = Simple1DCNN(1, 2)
model.load_state_dict(torch.load("./simple_nn_model.pth"))
model.eval()

# 테스트 데이터 불러오기
features = ["Delta", "Theta", "Alpha1", "Alpha2", "Beta1", "Beta2", "Gamma1", "Gamma2"]
test_data = pd.read_csv("./test_w_label.csv")
train_data = pd.read_csv("./train.csv")

# SubjectID를 정의, 예를 들어 1.0
subject_id = 5.0

# 특정 SubjectID로 train과 val 데이터 필터링
train_data = train_data[train_data["SubjectID"] == subject_id]
test_data = test_data[test_data["SubjectID"] == subject_id]

y_test = test_data["Label"]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
X_test = test_data[features]
X_test -= mean
X_test /= std

# 테스트 데이터를 텐서로 변환
X_test_tensor = torch.FloatTensor(X_test.values)
X_test_tensor = X_test_tensor.unsqueeze(1)

# 모델을 통과시켜 특징 추출
with torch.no_grad():
    feature_outputs = model(X_test_tensor)

# t-SNE 객체 생성
tsne = TSNE(n_components=2, random_state=0)

# 원본 데이터에 t-SNE 적용
tsne_obj_raw = tsne.fit_transform(X_test.values)

# 모델 출력에 t-SNE 적용
tsne_obj_model = tsne.fit_transform(feature_outputs.numpy())

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
axes[1].set_title("t-SNE on Model Output")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")
axes[1].legend(
    handles=scatter.legend_elements()[0], labels=set(y_test), title="Classes"
)

plt.tight_layout()
plt.show()
