import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

# =====================================
# 데이터 전처리
# =====================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: (x > 0.5).float()
    )
])

# MNIST 테스트 데이터
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# =====================================
# 모델 정의
# =====================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        # 입력층 → 은닉층
        self.fc1 = nn.Linear(784, 10)

        # 활성화 함수
        self.relu = nn.ReLU()

        # 은닉층 → 출력층
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):

        # (batch,1,28,28)
        # ↓
        # (batch,784)
        x = x.view(-1, 784)

        x = self.fc1(x)
        print(x[0].tolist())
        x = self.relu(x)
        print(x[0].tolist())
        x = self.fc2(x)
        print(x[0].tolist())

        return x


model = MLP()

model.load_state_dict(
    torch.load("reform_model.pth")
)

# =====================================
# 테스트
# =====================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    image, label = test_dataset[0]
    image = image.unsqueeze(0)

    outputs = model(image)

    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )

    plt.title(f"Label: {label}")
    plt.show()
