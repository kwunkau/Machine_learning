from NeuralNetwork import NeuralNetwork
import numpy as np

# 共三层神经网络；并分别有2,2,1个神经元
nn = NeuralNetwork([2, 2, 1], 'tanh')
# 非线性 异或
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
	print(i, nn.predict(i))
