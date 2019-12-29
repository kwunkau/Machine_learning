from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()
print(digits.data.shape)
# 1797:一共1797张图片；64:一张图片有64个特征向量（即像素点）


# 灰度化
pl.gray()
pl.matshow(digits.images[3])
pl.show()