import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

# 데이터 로딩
train_data = pd.read_csv('./train.csv')
val_data = pd.read_csv('./val.csv')


features = ['Delta', 'Theta', 'Alpha1', 'Alpha2', 'Beta1', 'Beta2', 'Gamma1', 'Gamma2']
X_train = train_data[features]
y_train = train_data['Label']
X_val = val_data[features]
y_val = val_data['Label']

'''
param_grid = {
    'n_estimators' : [100, 50],
    'min_samples_split': [100, 50, 10],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10]
}

grid_search = GridSearchCV(clf_rf, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
'''

# 분류기 학습
#grid_search.fit(X_train, y_train)
clf_rf = RandomForestClassifier(random_state=42, min_samples_leaf=2, min_samples_split=100, n_estimators=500)
clf_rf.fit(X_train, y_train)

# 학습 및 검증 데이터에 대한 예측
y_train_pred = clf_rf.predict(X_train)
y_val_pred = clf_rf.predict(X_val)

# 정확도 계산
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

'''
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_clf = grid_search.best_estimator_
y_val_pred = best_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")
'''

# 모델 저장
joblib.dump(clf_rf, './rf_model.pkl')
