import pickle
import numpy as np
from skimage.feature import local_binary_pattern, hog
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 定义函数用于读取数据
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

# 读取 CIFAR-100 数据集
batch = unpickle('cifar-100-python/train')
images = batch[b'data']

# 提取图像的 LBP 特征
lbp_features = []
for image in images:
    gray_image = cv2.cvtColor(image.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
    lbp_features.append(hist)
lbp_features = np.array(lbp_features)

# 对提取的特征做 PCA 降维
pca = PCA(n_components=50)
lbp_features_pca = pca.fit_transform(lbp_features)


# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(lbp_features_pca, batch[b'coarse_labels'][:len(lbp_features_pca)], test_size=0.2, random_state=42)

# 初始化并训练朴素贝叶斯分类器
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_accuracy_lbp = accuracy_score(y_test, nb_classifier.predict(X_test))
# 初始化并训练KNN分类器
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_accuracy_lbp = accuracy_score(y_test, knn_classifier.predict(X_test))
# 初始化并训练逻辑回归分类器

lr_classifier = LogisticRegression(max_iter=1000 ,solver='sag')
lr_classifier.fit(X_train, y_train)
lr_accuracy_lbp = accuracy_score(y_test, lr_classifier.predict(X_test))
# 提取图像的 HOG 特征
hog_features = []
for image in images:
    gray_image = cv2.cvtColor(image.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    hog_features.append(hog_feature)
hog_features = np.array(hog_features)

# 对提取的特征做 PCA 降维
hog_features_pca = pca.fit_transform(hog_features)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(hog_features_pca, batch[b'coarse_labels'][:len(hog_features_pca)], test_size=0.2, random_state=42)

# 初始化并训练朴素贝叶斯分类器
nb_classifier.fit(X_train, y_train)
nb_accuracy_hog = accuracy_score(y_test, nb_classifier.predict(X_test))
# 初始化并训练KNN分类器

knn_classifier.fit(X_train, y_train)
knn_accuracy_hog = accuracy_score(y_test, knn_classifier.predict(X_test))
# 初始化并训练逻辑回归分类器
lr_classifier.fit(X_train, y_train)
lr_accuracy_hog = accuracy_score(y_test, lr_classifier.predict(X_test))
# 提取图像的 SIFT 特征
sift_features = []
sift = cv2.SIFT_create()
for image in images:
    gray_image = cv2.cvtColor(image.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is not None:
        sift_features.append(descriptors.flatten())
    else:
        sift_features.append(np.zeros(128))
feature_length = len(sift_features[0])
sift_features = [feature for feature in sift_features if len(feature) == feature_length]
sift_features = np.array(sift_features)

# 对提取的特征做 PCA 降维
sift_features_pca = pca.fit_transform(sift_features)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(sift_features_pca, batch[b'coarse_labels'][:len(sift_features_pca)], test_size=0.2, random_state=42)

# 初始化并训练朴素贝叶斯分类器
nb_classifier.fit(X_train, y_train)
nb_accuracy_sift = accuracy_score(y_test, nb_classifier.predict(X_test))
# 初始化并训练KNN分类器
knn_classifier.fit(X_train, y_train)
knn_accuracy_sift = accuracy_score(y_test, knn_classifier.predict(X_test))
# 初始化并训练逻辑回归分类器
lr_classifier.fit(X_train, y_train)
lr_accuracy_sift = accuracy_score(y_test, lr_classifier.predict(X_test))
# 输出不同特征提取方法下的分类结果
print("LBP特征提取下朴素贝叶斯分类器准确率：", nb_accuracy_lbp)
print("HOG特征提取下朴素贝叶斯分类器准确率：", nb_accuracy_hog)
print("SIFT特征提取下朴素贝叶斯分类器准确率：", nb_accuracy_sift)
print("LBP特征提取下KNN分类器准确率：", knn_accuracy_lbp)
print("HOG特征提取下KNN分类器准确率：", knn_accuracy_hog)
print("SIFT特征提取下KNN分类器准确率：", knn_accuracy_sift)
print("LBP特征提取下逻辑回归分类器准确率：", lr_accuracy_lbp)
print("HOG特征提取下逻辑回归分类器准确率：", lr_accuracy_hog)
print("SIFT特征提取下逻辑回归分类器准确率：", lr_accuracy_sift)

best_nb_lbp = nb_classifier if nb_accuracy_lbp > max(nb_accuracy_hog, nb_accuracy_sift) else None
best_nb_hog = nb_classifier if nb_accuracy_hog > max(nb_accuracy_lbp, nb_accuracy_sift) else None
best_nb_sift = nb_classifier if nb_accuracy_sift > max(nb_accuracy_lbp, nb_accuracy_hog) else None
# 选择准确率最高的分类器
best_nb_classifier = best_nb_lbp or best_nb_hog or best_nb_sift
# 选择最佳KNN分类器
best_knn_lbp = knn_classifier if knn_accuracy_lbp > max(knn_accuracy_hog, knn_accuracy_sift) else None
best_knn_hog = knn_classifier if knn_accuracy_hog > max(knn_accuracy_lbp, knn_accuracy_sift) else None
best_knn_sift = knn_classifier if knn_accuracy_sift > max(knn_accuracy_lbp, knn_accuracy_hog) else None

# 根据准确率选择最佳的KNN分类器
best_knn_classifier = best_knn_lbp or best_knn_hog or best_knn_sift

# 选择最佳逻辑回归分类器
best_lr_lbp = lr_classifier if lr_accuracy_lbp > max(lr_accuracy_hog, lr_accuracy_sift) else None
best_lr_hog = lr_classifier if lr_accuracy_hog > max(lr_accuracy_lbp, lr_accuracy_sift) else None
best_lr_sift = lr_classifier if lr_accuracy_sift > max(lr_accuracy_lbp, lr_accuracy_hog) else None

# 根据准确率选择最佳的逻辑回归分类器
best_lr_classifier = best_lr_lbp or best_lr_hog or best_lr_sift

# 保存最佳朴素贝叶斯分类器

with open('best_nb_classifier.pkl', 'wb') as file:
    pickle.dump(best_nb_classifier, file)

with open('best_knn_classifier.pkl', 'wb') as file:
    pickle.dump(best_knn_classifier, file)

with open('best_lr_classifier.pkl', 'wb') as file:
    pickle.dump(best_lr_classifier, file)

