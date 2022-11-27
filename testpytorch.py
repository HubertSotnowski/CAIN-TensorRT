from cain.cain import CAIN
import torch
import os
f1 = torch.rand((1, 6, 256,854))
model = CAIN(3).cuda()
print(model(f1.cuda()).shape)
