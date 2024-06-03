import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple
import ptwt, pywt
from torch.cuda.amp import autocast
import warnings

class WaveletLayer(nn.Module):
    def __init__(self, 
                input_shape : list,  # C,H,W,D
                wavelet_type : pywt._extensions._pywt.Wavelet, 
                wavelet_level : int = None,
                conv_in_axes : int = -4 # C
                ):
        super().__init__()

        self.wavelet_type = wavelet_type

        self.key = list()


    @autocast(enabled= False)
    def forward(self, x):
        # B, C, W, H, D
        # [tensor | dict]
        self.wavelet_input_shape = x.shape
        x_wavelet = ptwt.wavedec3(x, self.wavelet_type, level= 1)

        tmp_x = None
        for i, wavelet in enumerate(x_wavelet):
            if i == 0: 
                tmp_x = wavelet 
                continue
            for k, v in wavelet.items():
                self.key.append(k)
                tmp_x = torch.cat((tmp_x, v), -4) if tmp_x != None else v
        # B, 8C, W, H, D
        return tmp_x


    def inverse(self, x):
        # B, 8C, W, H, D -> B, C, W, H, D
        print(x.shape)
        batches = torch.chunk(x, 8, -4)
        print(len(batches))
        self.wavelet_0 = batches[0]
        # create dict
        x_dict = dict()
        for i, key in enumerate(self.key):
            print(i, key)
            x_dict[key] =  batches[i + 1]
        reconstruction = ptwt.waverec3([self.wavelet_0, x_dict], self.wavelet_type)

        # check padding 
        reconstruction = reconstruction[tuple(slice(s) for s in self.wavelet_input_shape)]

        return reconstruction

if __name__ == "__main__":
    a = WaveletLayer((15, 40, 56, 40), pywt.Wavelet("sym4"), conv_in_axes=-2).to("cuda")
    x = torch.randn(3, 15, 40, 56, 40).to("cuda")
    # print(x.shape[1,2])
    print(a(x).shape)
