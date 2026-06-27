# Minecraft MNIST

**Kiwoo** | Minecraft MNIST (to learn Logical Redstone and PyTorch)

Date : **May04-Jun14 2026** (42days)

## 정보

![img2](Image\img2.png)

**Minecraft Version** : 26.1.2

**Python Version** : 3.14.5

**Python Library Version** : 
|Library            | Version       |
|-                  |-              |
| contourp          | 1.3.3         |
| cycler            | 0.12.1        |
| filelock          | 3.29.0        |
| fonttools         | 4.63.0        |
| fsspec            | 2026.4.0      |
| immutable-views   | 0.6.1         |
| Jinja2            | 3.1.6         |
| kiwisolver        | 1.5.0         |
| MarkupSafe        | 3.0.3         |
| matplotlib        | 3.10.9        |
| mcschematic       | 11.4.4        |
| mpmath            | 1.3.0         |
| nbtlib            | 2.0.4         |
| networkx          | 3.6.1         |
| numpy             | 2.4.6         |
| packaging         | 26.2          |
| pillow            | 12.2.0        |
| pyparsing         | 3.3.2         |
| python-dateutil   | 2.9.0.post0   |
| setuptools        | 81.0.0        |
| six               | 1.17.0        |
| sympy             | 1.14.0        |
| torch             | 2.12.0        |
| torchvision       | 0.27.0        |
| typing_extensions | 4.15.0        |

## 개발동기

[![마인크래프트로 인게임에서 실제 작동하는 AI 회로 만들기](https://img.youtube.com/vi/-kZxh5RTWzg/maxresdefault.jpg)](https://www.youtube.com/watch?v=-kZxh5RTWzg)
[마인크래프트로 인게임에서 실제 작동하는 AI 회로 만들기 - 치즈 사하야담]

- 이 영상을 보고 모드가 아니라 바닐라로 만들어 봐야겠다는 생각이 들어서 시작함

## 모델

### Minecraft MNIST Model

**Modle** : Multi Layer Perceptron, MLP

**Accuracy** : 90.28%

**Speed** : 747 Redstone Tick (74.7s *Ideal* )

**Input** : 28x28 MNIST Image (Monochrome)

**Hidden Layer** (fc1) : 1 Layer 10 Neurons
- Parameter : 7,840 (28x28 x10) weight, 10 bias (4bit integer -8 ~ 7, used HEX)
- Activation Function : ReLU

**Output** (fc2) : 10 Neurons (0-9)
- Parameter : 100 (10 x 10) weight, 10 bias (4bit unsigned integer, 1bit sign -15 ~ 15)
- Activation Function : None

---

### 상세

- 형태가 단순하고 연산량을 줄일 수 있도록 일반적인 MNIST모델과 다르게 CNN이 아닌 MLP를 이용했습니다.
- 마인크래프트 레드스톤으로 표현하기 쉽게 회색조 이미지가 아닌 흑백 이미지를 입력값으로 받도록 했습니다.
- 회색조가 아닌 흑백 이미지를 사용하므로써 첫번째 레이어에 곱셈 연산을 단순화시킬 수 있었습니다. (w * 0 = 0, w * 1 = w)
- 회로를 최대한 단순화하기 위해 은닉층을 1층으로, 은닉층 뉴런을 10개로 설정했습니다.
- 파라미터값을 효과적으로 저장하기 위해 레드스톤 신호강도를 이용해 4bit 값을 1HEX 값으로 저장했습니다.
- HEX 값은 후에 덧셈연산을 할 때 2진수로 변환하였습니다.
- 활성화 함수는 구현이 쉬운 ReLU를 이용했습니다.
- 두번째 레이어에서는 곱셈기를 효율적으로 구성하기 위해 부호와 크기를 따로 저장을 해두었습니다.
- 덧셈을 할 때는 1의 보수법인 상태로 더한 후 더해진 음수의 개수만큼 1을 다시 더해 값을 보정하는 방식을 사용했습니다.

### 구조

![img1](Image\img1.png)

- **Input**
    - **Input Pad** (Gray)
    - **Transmission** (Light Blue)

- **Display**
    - **Input Signal Processor** (Orange)
    - **Input Image Display** (Left Lime)
    - **Model Input** (Right Lime)
    - **Output Number Display** (Pink)
    - **Loading Bar Display** (Bottom Orange)

- **fc1**
    - **fc1 Parameter ROM** (Red)
    - **Multiplier** (Blue)
    - **Hex to Bin** (Yellow)
    - **Accumulator and ReLU** (Green)

- **fc2**
    - **fc2 Parameter and Correction Constant ROM and Multiplier** (Magenta)
    - **Accumulator** (Cyan, Green)

- **Comparator**
    - **Comparator** (Purple)
    - **Max Number Selector** (Yellow)

## 과정

### 파라미터 이산화

![fc1](Image\fc1.png) ![fc2](Image\fc2.png)

- 파이토치로 만든 모델의 파라미터 종류를 16가지 이하의 값으로 이산화 했을 때 정확도가 90%가 넘었습니다.

- 파라미터 종류가 16개를 넘지 않기 때문에 레드스톤 신호 세기로도 연산을 구현할 수 있게 되었습니다.

- fc1의 경우 HEX로 변환해 값을 적용했고, fc2의 경우 4bit의 크기값과 1bit의 부호값을 따로 저장했습니다.

### Accumulator

- 연산된 수들을 다 더하기 위해 Automatic Accumulator를 사용했습니다.
- Accumulator의 정확한 구동법을 제대로 이해하지 못해 제작 과정에 어려움이 있었습니다.
- Accumulator는 정확히 2tick 간격의 입력이 들어와야 제대로 연산이 되는데 이를 잘 알지 못해 값에 오차가 생기는 문제가 있었습니다.

![img3](Image\img3.png)

- 2tick 간격으로 1을 세 번 입력 했을 때 결과값이 3(11)으로 제대로 나옵니다.

![img4](Image\img4.png)

- 반면 3tick 간격으로 1을 세번 입력했을 때 4(100)로 값에 오차가 생기는 것을 볼 수 있습니다.

## 참고

[![I Made an AI with just Redstone!](https://img.youtube.com/vi/DQ0lCm0J3PM/maxresdefault.jpg)](https://www.youtube.com/watch?v=DQ0lCm0J3PM)
[I Made an AI with just Redstone! - mattbatwings]

- 제작 방식에 있어 유튜버 mattbatwings의 영상을 참고했습니다.

