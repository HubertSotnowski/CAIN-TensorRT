from cain.cain import CAIN
import torch
import os
f1 = torch.rand((1, 3, 256,854*2))
model = CAIN(3).cuda()
model(f1.cuda())
