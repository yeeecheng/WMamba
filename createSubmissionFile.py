import os
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description= "rename predict files")
    parse.add_argument("-s", "--src_path", type= str, help= "origin folder path")
    parse.add_argument("-d", "--dst_path", type= str, help= "save folder destination")
    args = parse.parse_args()
    return args

def rename_file(src_path, dst_path):


    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)

    special_file = ["180", "189", "287", "318", "369"]
    num = 1
    for file in os.listdir(src_path):
        if file.split(".")[-1] == "gz":
            new_file_name = file.split(".")[0].split("_")[1][1:4]   
            
            if new_file_name in special_file:
                
                if new_file_name == "369":
                    new_file_name += "_1"
                else:
                    new_file_name += "_0"
            os.rename(os.path.join(src_path, file), os.path.join(dst_path, new_file_name + ".nii.gz"))
            print("[{:02}/50] Convert file from {} to {}".format(num, os.path.join(src_path, file), os.path.join(dst_path, new_file_name + ".nii.gz")))
            num += 1
if __name__ == "__main__":
    args = parse_args()
    rename_file(args.src_path, args.dst_path)