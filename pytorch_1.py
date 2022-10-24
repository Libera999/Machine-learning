import torch
import numpy as np

x = torch.rand(2,2)

a=np.ones(5)
b=torch.from_numpy(a)

a+=1
print(b)

