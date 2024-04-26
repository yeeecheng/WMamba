other_file=`ls ./W-Mamba/*.py`
trainer=`find ./W-Mamba/ -name "nnUNetTrainer*"  -type f`
net=`find ./W-Mamba/ -name "WMamba*"  -type f`

echo "Save WMamba"
echo "CP File From U-Mamba to W-Mamba"
echo "Trainer:"
for i in $trainer;
do
    file_name=`basename $i`
    folder_path=`dirname $i`
    echo "  $file_name"
    cp ./U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/$file_name $folder_path
done

echo "Nets:"
for i in $net;
do
    file_name=`basename $i`
    folder_path=`dirname $i`
    echo "  $file_name"
    cp ./U-Mamba/umamba/nnunetv2/nets/$file_name $folder_path
done

echo "Others:"
for i in $other_file;
do
    file_name=`basename $i`
    folder_path=`dirname $i`
    echo "  $file_name"
    cp ./U-Mamba/umamba/nnunetv2/$file_name $folder_path
done
