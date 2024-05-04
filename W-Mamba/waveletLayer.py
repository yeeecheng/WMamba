import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple
import ptwt, pywt
from torch.cuda.amp import autocast

class WaveletLayer(nn.Module):
    def __init__(self, 
                 input_shape : list,  # C,H,W,D
                 wevelet_type : pywt._extensions._pywt.Wavelet, 
                 wavelet_level : int = None,
                 conv_in_axes : int = -4 # C
                 ):
        
        
        super().__init__()

        self.wavelet_type = wevelet_type

        wavelet_input_shape = list(input_shape)
        wavelet_input_shape[-4], wavelet_input_shape[conv_in_axes] = wavelet_input_shape[conv_in_axes], wavelet_input_shape[-4]
        self.wavelet_level = wavelet_level if wavelet_level != None else pywt.dwtn_max_level(wavelet_input_shape, wevelet_type)

        assert self.wavelet_level != 0

        self.conv_in_axix = conv_in_axes
        self.extract_in_level_layer = [
                                nn.Sequential(
                                    nn.Conv3d(
                                        in_channels = input_shape[conv_in_axes]*7,
                                        out_channels= input_shape[conv_in_axes],
                                        kernel_size= (3,3,3),
                                        padding = "same",
                                    ),
                                    nn.LeakyReLU()
                                ) for i in range(self.wavelet_level)
                            ]
        self.extract_in_level_layer = nn.ModuleList(self.extract_in_level_layer)
        self.extract_all_level_layer = nn.Sequential(
                                        nn.Conv3d(
                                            in_channels = input_shape[conv_in_axes] * (self.wavelet_level+1),
                                            out_channels= input_shape[conv_in_axes],
                                            padding= "same",
                                            kernel_size= (3,3,3)
                                        ),
                                        nn.LeakyReLU()
                                    )
                                
    @autocast(enabled=False)
    def forward(self, x):
        # B,A,B,X,C => B,X,A,B,C
        x = x.transpose(self.conv_in_axix, -4)
        # B,X,A,B,C => [L:N]B,X,A',B',C'
        x_wavelet = ptwt.conv_transform_3.wavedec3(x, self.wavelet_type, level = self.wavelet_level)

        # get the maximum shape H,W,D (level 0)
        # B,X,MA,MB,MC
        max_shape = x_wavelet[-1]['aad'].shape[-3:]
        # upsample largest level 'aaa' 
        # B,X,A',B',C' => B,X,MA,MB,MC
        in_layer_upsample = torch.nn.Upsample(max_shape)
        x_wavelet_result = in_layer_upsample(x_wavelet[0])
        
        # iterate all level, except largest one
        for i , wavelet in enumerate(x_wavelet):
            if i == 0 : continue
            tmp_x = None
            # concat and upsample the all feq in one level
            for k, v in wavelet.items():
                # B,X,A',B',C' => B,X,MA,MB,MC
                v = in_layer_upsample(v)
                tmp_x = torch.cat((tmp_x, v) , -4)  if tmp_x != None else v
            
            # extract the feature in one level
            # B,X*7,MA,MB,MC => B,X,MA,MB,MC
            tmp_x = self.extract_in_level_layer[i-1](tmp_x)
            # concat into result
            x_wavelet_result = torch.cat((x_wavelet_result, tmp_x) , -4)
        
        # upsample into origin size
        # B,X*L,MA,MB,MC => B,X*L,A,B,C
        x_wavelet_result = torch.nn.Upsample(x.shape[-3:])(x_wavelet_result)
        # extract the feature with all  level
        out = self.extract_all_level_layer(x_wavelet_result)
        # B,X,A,B,C => B,A,B,X,C
        out = out.transpose(self.conv_in_axix, -4)
        return out

if __name__ == "__main__":
    a = WaveletLayer((15, 40, 56, 40), pywt.Wavelet("sym4"), conv_in_axes=-2).to("cuda")
    x = torch.randn(3, 15, 40, 56, 40).to("cuda")
    # print(x.shape[1,2])
    print(a(x).shape)

