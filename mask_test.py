import numpy as np
import matplotlib.pyplot as plt
import torch

np_mask = np.load('/Users/mattbendel/Desktop/lolcat_extra.npy')
print(np_mask.shape)
loaded = torch.from_numpy(np_mask).reshape(-1)
print(loaded.shape)

m = np.zeros((384, 384))

a = [1, 23, 42, 60, 77, 92, 105, 117, 128, 138, 147, 155, 162, 169, 176, 182, 184, 185, 186, 187, 188, 189, 190,
     191, 192, 193, 194, 195,
     196, 197, 198, 199, 200, 204, 210, 217, 224, 231, 239, 248, 258, 269, 281, 294, 309, 326, 344, 363]

a = np.array(a)
m[:, a] = 1

mask = torch.from_numpy(m).reshape(-1)
missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 2
print(missing_r)
plt.imshow(np_mask, cmap='gray')
plt.savefig('mask_test.png')