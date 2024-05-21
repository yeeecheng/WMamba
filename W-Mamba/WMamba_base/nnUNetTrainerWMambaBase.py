from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
import torch
import os

if os.environ.get("ipynb") == None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm as tqdm



class nnUNetTrainerWMambaBase(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if os.environ.get('training_epochs') == None:
            raise ValueError("Epochs environment variable not define")
        
        if os.environ.get('checkpoint_save_every') != None:
            self.save_every = int(os.environ.get('checkpoint_save_every'))
        
        self.num_epochs = int(os.environ.get('training_epochs'))
        print("Total Epochs ", self.num_epochs)
        print("Checkpoint Save Every ", self.save_every)
    
    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in tqdm(range(self.num_iterations_per_epoch)):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
