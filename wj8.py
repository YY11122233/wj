import streamlit as st
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
from skimage.feature import local_binary_pattern, hog
from PIL import Image
from sklearn.decomposition import PCA
import torchvision.transforms as transforms

# 定义神经网络模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 100)  # CIFAR-100有100个类别

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 加载实验一中的模型
with open('best_nb_classifier.pkl', 'rb') as file:
    nb_classifier = pickle.load(file)
with open('best_knn_classifier.pkl', 'rb') as file:
    knn_classifier = pickle.load(file)
with open('best_lr_classifier.pkl', 'rb') as file:
    lr_classifier = pickle.load(file)

# 加载实验二中的神经网络模型
model = CustomModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
# 定义特征提取函数
def extract_features(image, method='LBP'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'LBP':
        lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
        return hist
    elif method == 'HOG':
        hog_feature = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        return hog_feature
    elif method == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        if descriptors is not None:
            return descriptors.flatten()
        else:
            return np.zeros(128)


with open('cifar-100-python/meta', 'rb') as file:
    # 使用pickle模块加载meta文件中的数据
    label_data = pickle.load(file, encoding='bytes')

# 获取细粒度标签名称列表
fine_label_names = [label.decode('utf-8') for label in label_data[b'fine_label_names']]



# 定义函数将索引转换为对应的标签
def index_to_label(index):
    return fine_label_names[index]

# 定义函数使用神经网络模型进行分类
def classify_image_nn(image, model_type='NN'):
    model_type == 'NN'
    image_pil = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    return index_to_label(predicted.item())
# 定义函数将图像和特征提取方法组合使用传统分类器进行分类
def classify_image_traditional(image, model_type, feature_type):
        image_resized = cv2.resize(image, (32, 32))
        features = extract_features(image_resized, method=feature_type)
        n_features_expected = 50
        if features.shape[0] > n_features_expected:
            features = features[:n_features_expected]
        else:
            features_pca = features.reshape(1, -1)
        if model_type == 'NB':
            classifier = nb_classifier
        elif model_type == 'KNN':
            classifier = knn_classifier
        elif model_type == 'LR':
            classifier = lr_classifier
        result_index = classifier.predict(features.reshape(1, -1))
        return index_to_label(result_index[0])

# Streamlit应用
st.title('Streamlit图像分类应用')
st.title('2109120119_晏叶婷')
uploaded_file = st.file_uploader('上传图像文件', type=['jpg', 'jpeg', 'png'])
result = None  # 初始化结果变量
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='上传的图像', use_column_width=True)

    if not st.checkbox('使用神经网络模型（NN）'):
        model_type = st.selectbox('选择分类器', ['NB', 'KNN', 'LR'])
        feature_type = st.selectbox('选择特征提取方法', ['LBP', 'HOG', 'SIFT'])

        if st.button('分类'):
            result = classify_image_traditional(image_rgb, model_type=model_type, feature_type=feature_type)
    else:
        if st.button('分类'):
            result = classify_image_nn(image_rgb)

    if result is not None:
        st.write(f'分类结果：{result}')