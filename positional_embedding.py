import torch
import torch.nn as nn
import math


def SinusoidalPositionalEncoding(embed_dim: int, height: int, width: int, cls_token: bool = False):

    """
    Genera codifiche posizionali sinusoidali 2D.

    La codifica è simile a quella utilizzata in DETR. Per ogni posizione (h, w),
    il vettore di codifica pe[h,w] di dimensione embed_dim è popolato come segue:
    Sia embed_dim_half = embed_dim / 2.
    Per i primi embed_dim_half canali (relativi alla larghezza):
        pe[h,w,2i]   = sin(w / (10000^(2i/embed_dim_half)))
        pe[h,w,2i+1] = cos(w / (10000^(2i/embed_dim_half)))
    Per i successivi embed_dim_half canali (relativi all'altezza):
        pe[h,w,embed_dim_half + 2i]   = sin(h / (10000^(2i/embed_dim_half)))
        pe[h,w,embed_dim_half + 2i+1] = cos(h / (10000^(2i/embed_dim_half)))

    La funzione restituisce un tensore di forma (height*width, embed_dim) o
    (1 + height*width, embed_dim) se cls_token è True.

    :param embed_dim: La dimensione degli embedding del modello. Deve essere un multiplo di 4.
    :param height: L'altezza della griglia 2D.
    :param width: La larghezza della griglia 2D.
    :param cls_token: Se True, una codifica posizionale nulla (zeri) per un token CLS
                    viene anteposta alla sequenza delle codifiche posizionali.
                    Default: False.
    :return: Un tensore contenente le codifiche posizionali.
            Forma: (height * width, embed_dim) se cls_token è False.
            Forma: (1 + height * width, embed_dim) se cls_token è True,
                    dove la prima riga è il vettore nullo per il token CLS.
    :raises ValueError: Se embed_dim non è un multiplo di 4.
    """
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