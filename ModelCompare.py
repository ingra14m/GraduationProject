import torch
import torch.nn as nn

import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.layer1 = nn.Linear(2000, 500)
        self.layer2 = nn.Linear(500, 50)
        self.layer3 = nn.Linear(50, 10)

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        return h


inpput = torch.randn((256, 2000), device=device)
model = BaseModel()
model.to(device)

start = time.time()
model(inpput)

print(time.time() - start)