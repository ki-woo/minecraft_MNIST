import mcschematic
from barrel import block0, block1
import torch

weights = torch.load("reform_model.pth")

for name, param in weights.items():
    weights[name] = torch.round(param).to(torch.int16)

def generate_schem():
    schem = mcschematic.MCSchematic()

    for i in range(10):
        w = weights["fc2.weight"][i].tolist()
        print(w)
        
        for j in range(10):
            if w[j] < 0:
                schem.setBlock((-i*16 -5, -23, -j*6 + 3), block1)
                w[j] *= -1
            else:
                schem.setBlock((-i*16 -5, -23, -j*6 + 3), block0)
            
            # 1
            if w[j] % 16 >= 8:
                for k in range(8):
                    schem.setBlock((-i*16 + 6, -k*2 - 4, -j*6), block1)
            else:
                for k in range(8):
                    schem.setBlock((-i*16 + 6, -k*2 - 4, -j*6), block0)
            
            # 2
            if w[j] % 8 >= 4:
                for k in range(8):
                    schem.setBlock((-i*16 + 4, -k*2 - 3, -j*6), block1)
            else:
                for k in range(8):
                    schem.setBlock((-i*16 + 4, -k*2 - 3, -j*6), block0)
            
            # 4
            if w[j] % 4 >= 2:
                for k in range(8):
                    schem.setBlock((-i*16 + 2, -k*2 - 2, -j*6), block1)
            else:
                for k in range(8):
                    schem.setBlock((-i*16 + 2, -k*2 - 2, -j*6), block0)
            
            # 8
            if w[j] % 2 >= 1:
                for k in range(8):
                    schem.setBlock((-i*16, -k*2 - 1, -j*6), block1)
            else:
                for k in range(8):
                    schem.setBlock((-i*16, -k*2 - 1, -j*6), block0)


    # schem.save(
    #     outputFolderPath="schematics/output",
    #     schemName="fc2.weight",
    #     version=mcschematic.Version.JE_1_21_4
    # )

generate_schem()

def negaNum():
    arr = []
    for i in range(10):
        n = 0
        w = weights["fc2.weight"][i].tolist()
        b = weights["fc2.bias"][i]

        for j in w:
            if j < 0:
                n += 1
        
        if b < 0:
            n += 1
        
        arr.append(n * 3)
    
    return arr


#print(weights["fc2.bias"].tolist())

#print(negaNum())
