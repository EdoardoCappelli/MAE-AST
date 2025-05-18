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
# print("Forma dell'output di Conv2d:", output_conv.shape)
# print("Output di Conv2d:\n", output_conv)

# N, C_out, H_out, W_out = output_conv.shape
# total_patches = H_out * W_out # Numero totale di patch spaziali nell'output

# print(f"Numero totale di patch (output features spaziali): {total_patches} ({H_out}x{W_out})")

# # 1. Calcola il numero di patch da mascherare (obiettivo: 75%)
# percentuale_mascheramento = 0.75
# # Arrotonda al numero intero più vicino di patch da mascherare
# num_masked_patches = int(round(total_patches * percentuale_mascheramento))

# all_patch_indices = np.arange(total_patches)

# # 3. Mescola gli indici in modo casuale
# # Per la riproducibilità, è possibile impostare un seed prima di questa operazione:
# # np.random.seed(42) # Esempio di seed
# np.random.shuffle(all_patch_indices)

# # 4. Definisci masked_idx e retained_idx
# # Gli indici mascherati sono i primi 'num_masked_patches' dall'array mescolato
# # Vengono convertiti in lista e ordinati per una visualizzazione più chiara
# masked_idx = sorted(list(all_patch_indices[:num_masked_patches]))

# # Gli indici mantenuti (non mascherati) sono i restanti
# retained_idx = sorted(list(all_patch_indices[num_masked_patches:]))

# num_retained_patches = total_patches - num_masked_patches

# print(f"Numero di patch da mascherare (num_masked_patches): {num_masked_patches} su {total_patches}")
# print(f"Numero di patch mantenute: {num_retained_patches} su {total_patches}")

# print(f"Indici delle patch mascherate (masked_idx, {len(masked_idx)} elementi): {masked_idx}")
# print(f"Indici delle patch mantenute (retained_idx, {len(retained_idx)} elementi): {retained_idx}")

# print(output_conv[0])

# # retained_patches = output_conv[]