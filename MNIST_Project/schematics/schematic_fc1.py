import mcschematic
from barrel import block
import torch

weights = torch.load("reform_model.pth")

for name, param in weights.items():
    weights[name] = torch.round(param).to(torch.int16)

def generate_schem():
    for i in range(10):
        schem = mcschematic.MCSchematic()
        w = weights["fc1.weight"][i].tolist()

        for j in range(4):
            for k in range(7):
                for l in range(28):
                    if k % 2 == 0:
                        schem.setBlock((0, -j*14 -k*2 - 1, l*2 + 2), block[w[(j*7 + k) * 28 + l]])
                    else:
                        schem.setBlock((0, -j*14 -k*2 - 1, l*2), block[w[(j*7 + k) * 28 + l]])

        schem.save(
            outputFolderPath="schematics/output",
            schemName="fc1.weight"+str(i+1),
            version=mcschematic.Version.JE_1_21_4
        )

# generate_schem()

# print(weights["fc2.bias"].tolist())
