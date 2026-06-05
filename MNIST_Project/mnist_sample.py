import torch
import torch.nn as nn
import torch.optim as optim

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

# MNIST 학습 데이터
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# MNIST 테스트 데이터
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 배치 단위로 데이터를 불러오기
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
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


# =====================================
# 손실 함수
# =====================================

criterion = nn.CrossEntropyLoss()


# =====================================
# 옵티마이저
# =====================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================================
# 학습
# =====================================

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss = {total_loss:.4f}"
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

    for param in model.parameters():
        param.copy_(torch.round(param * 5) / 5)
    
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

target_accuracy = 90

if accuracy >= target_accuracy:

    torch.save(
        model.state_dict(),
        "best_model.pth"
    )

    print("저장 완료")
