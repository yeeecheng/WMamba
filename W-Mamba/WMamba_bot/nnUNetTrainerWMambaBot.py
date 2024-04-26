from nnunetv2.training.nnUNetTrainer.nnUNetTrainerWMambaBase import nnUNetTrainerWMambaBase
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
import os
from nnunetv2.nets.WMambaBot_3d import get_wmamba_bot_3d_from_plans
from tqdm import tqdm



class nnUNetTrainerWMambaBot(nnUNetTrainerWMambaBase):

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        
        model = get_wmamba_bot_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                          num_input_channels, deep_supervision=enable_deep_supervision)
        
        
        print("WMambaBot")

        return model
