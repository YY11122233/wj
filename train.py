import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor


# 定义神经网络模型
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # 定义神经网络结构
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 100)  # CIFAR-100有100个类别

    def forward(self, x):
        # 前向传播逻辑
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

# 加载CIFAR-100数据集
train_dataset = CIFAR100(root='./data', train=True, transform=ToTensor(), download=True)
test_dataset = CIFAR100(root='./data', train=False, transform=ToTensor(), download=True)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
model = CustomModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
best_accuracy = 0.0
for epoch in range(8):  # 假设训练8个epoch
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()

        # 检查标签是否在有效范围内
        labels = torch.clamp(labels, 0, 99)  # 将标签限制在0到99之间

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # 保存最佳模型参数
        torch.save(model.state_dict(), 'best_model.pth')

    print(f'Epoch {epoch+1}, Accuracy: {accuracy}')

print(f'Best Accuracy: {best_accuracy}')
