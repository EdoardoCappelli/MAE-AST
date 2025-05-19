import torch
import torch.nn as nn
import numpy as np

# Parametri comuni per il confronto
kernel_size = (3,3)
stride = (3,3)
padding = 0

input_data_np = np.arange(0, 100).reshape(10, 10)
input_tensor = torch.tensor(input_data_np, dtype=torch.float32)

input_tensor_unsqueezed = input_tensor.unsqueeze(0).unsqueeze(0)

unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)


output = unfold(input_tensor_unsqueezed)
print("Unfold output shape:", output.shape)

print(output[0])

# trasponiamo l'output in modo che ogni patch sia una riga
output = output.transpose(1, 2)

print("First patch of Unfold output:\n", output[0][0].reshape(3,3))

encoder_mask_emb = nn.Parameter(torch.FloatTensor(768).uniform_())
print("Encoder mask embedding shape:", encoder_mask_emb.shape)
print("Encoder mask embedding:\n", encoder_mask_emb)