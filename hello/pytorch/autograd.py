import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad = True)
print(a)

b = torch.sin(a)
# plt.plot(a.detach(), b.detach()) # WORKS FINE
# plt.plot(a, b) # NOT SUPPORT
# plt.show()

c = 2 * b
d = c + 1

out = d.sum()

out.backward()
print(a.grad)
plt.plot(a.detach(), a.grad.detach())
plt.show()