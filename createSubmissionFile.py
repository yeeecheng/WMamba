import os

file_path = "./pred"
res_path = "./test"

if not os.path.isdir(res_path):
    os.mkdir(res_path)

special_file = ["180", "189", "287", "318", "369"]

for file in os.listdir(file_path):
    if file.split(".")[-1] == "gz":
        new_file_name = file.split(".")[0].split("_")[1][1:4]   
        
        if new_file_name in special_file:
            
            if new_file_name == "369":
                new_file_name += "_1"
            else:
                new_file_name += "_0"


        os.rename(os.path.join(file_path, file), os.path.join(res_path, new_file_name + ".nii.gz"))

