import torch
import torch.nn as nn
import math

def SinusoidalPositionalEncoding(embed_dim: int, height: int, width: int, cls_token: bool = False):

    if embed_dim % 4 != 0:
        raise ValueError(
            "Impossibile utilizzare la codifica posizionale sin/cos con embed_dim={:d}. "
            "embed_dim deve essere un multiplo di 4, in modo che embed_dim/2 sia pari.".format(embed_dim)
        )

    pe_spatial = torch.zeros(embed_dim, height, width)

    embed_dim_half = embed_dim // 2 

    div_term = torch.exp(torch.arange(0., embed_dim_half, 2) * -(math.log(10000.0) / embed_dim_half))

    pos_w = torch.arange(0., width).unsqueeze(1)    # Forma: (width, 1)
    pos_h = torch.arange(0., height).unsqueeze(1)  # Forma: (height, 1)

    sin_enc_w = torch.sin(pos_w * div_term.unsqueeze(0)).transpose(0, 1)
    cos_enc_w = torch.cos(pos_w * div_term.unsqueeze(0)).transpose(0, 1)
    
    pe_spatial[0:embed_dim_half:2, :, :] = sin_enc_w.unsqueeze(1).repeat(1, height, 1)
    pe_spatial[1:embed_dim_half:2, :, :] = cos_enc_w.unsqueeze(1).repeat(1, height, 1)

    sin_enc_h = torch.sin(pos_h * div_term.unsqueeze(0)).transpose(0, 1)
    cos_enc_h = torch.cos(pos_h * div_term.unsqueeze(0)).transpose(0, 1)

    pe_spatial[embed_dim_half::2, :, :] = sin_enc_h.unsqueeze(2).repeat(1, 1, width)
    pe_spatial[embed_dim_half + 1::2, :, :] = cos_enc_h.unsqueeze(2).repeat(1, 1, width)
    
    pe_sequence = pe_spatial.contiguous().view(embed_dim, -1).transpose(0, 1)

    if cls_token:
        cls_token_pe = torch.zeros(1, embed_dim, device=pe_sequence.device, dtype=pe_sequence.dtype)
        final_pe = torch.cat((cls_token_pe, pe_sequence), dim=0)
        return final_pe.unsqueeze(0)
    else:
        return pe_sequence.unsqueeze(0)
