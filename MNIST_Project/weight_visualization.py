import torch
import matplotlib.pyplot as plt
import numpy as np

weights = torch.load("best_model.pth")

def input():
    w = weights["fc1.weight"].numpy()

    importance = np.mean(np.abs(w), axis=0)

    importance = importance.reshape(28, 28)

    # print(w.maximum)

    plt.imshow(importance)
    plt.colorbar()
    plt.title("Average Weight Magnitude")
    plt.show()

# ===================================

def neurons():
    fig, axes = plt.subplots(2, 5)

    for i in range(10):

        w = weights["fc1.weight"][i].numpy()

        axes[i//5, i%5].imshow(
            w.reshape(28, 28)
        )

        axes[i//5, i%5].set_title(
            f"Neuron {i}"
        )

        axes[i//5, i%5].axis("off")

    plt.tight_layout()
    plt.show()

# ===================================

def histo():
    w = weights["fc2.weight"].numpy().flatten()
    w *= 5

    # w = np.round(w).astype(int)


    plt.hist(w, bins=50)

    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.title("fc2.weight Distribution")

    plt.show()

histo()
