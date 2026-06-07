import torch
import torch.nn as nn

weights = torch.load(
    "best_model.pth"
)

for name, param in weights.items():
    weights[name] = torch.round(param * 5)

torch.save(
    weights,
    "reform_model.pth"
)
