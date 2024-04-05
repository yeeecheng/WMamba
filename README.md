untar
```
tar xvf {}
```
dataset convert
```
nnUNetv2_convert_MSD_dataset -i {}
```
dataset preprocess
```bash
nnUNetv2_plan_and_preprocess -d {}
nnUNetv2_plan_and_preprocess -d 4
```
train
```bash
nnUNetv2_train {} 3d_fullres all -tr {}
nnUNetv2_train 4 3d_fullres all -tr nnUNetTrainerUMambaBotT
```