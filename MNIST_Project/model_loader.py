import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

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

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
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
        x = self.relu(x)
        x = self.fc2(x)

        return x


model = MLP()

model.load_state_dict(
    torch.load("reform_model.pth")
)

# =====================================
# 정확도 측정
# =====================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        prediction = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            prediction == labels
        ).sum().item()

    accuracy = 100 * correct / total

    print(f"Accuracy : {accuracy:.2f}%")

