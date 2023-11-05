# Create a .npz file
import numpy as np
import cebra
from cebra import CEBRA
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from cebra import helper

# 임의의 정수 배열을 생성
test_array = np.array([1, 2, 3], dtype="int64")

# _is_integer 함수가 True를 반환하는지 확인
print(helper._is_integer(test_array))  # True를 출력해야 함

# timesteps = 5000
# neurons = 50
# out_dim = 8

# neural_data = np.random.normal(0, 1, (timesteps, neurons))
# continuous_label = np.random.normal(0, 1, (timesteps, 3))
# discrete_label = np.random.randint(0, 10, (timesteps,))

# single_cebra_model = cebra.CEBRA(
#     batch_size=512, output_dimension=out_dim, max_iterations=10, max_adapt_iterations=10
# )

# single_cebra_model.fit(neural_data, discrete_label)

# embedding = single_cebra_model.transform(neural_data)
# assert embedding.shape == (timesteps, out_dim)

# cebra.plot_embedding(embedding)
# print("plot end")

# print(embedding.shape)

# # plt.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2])
# plt.show()
