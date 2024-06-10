other_file=`ls ./W-Mamba/*.py`
trainer=`find ./W-Mamba/ -name "nnUNetTrainer*"  -type f`
net=`find ./W-Mamba/ -name "WMamba*"  -type f`

echo "Setup WMamba"
echo "CP File From W-Mamba to U-Mamba"
echo "Trainer:"
for i in $trainer;
do
    file_name=`basename $i`
    echo "  $file_name"
    cp $i ./U-Mamba/umamba/nnunetv2/training/nnUNetTrainer/
done

echo "Nets:"
for i in $net;
do
    file_name=`basename $i`
    echo "  $file_name"
    cp $i ./U-Mamba/umamba/nnunetv2/nets/ 
done

echo "Others:"
for i in $other_file;
do
    file_name=`basename $i`
    echo "  $file_name"
    cp $i ./U-Mamba/umamba/nnunetv2/
done

cp evaluate_predictions.py ./U-Mamba/umamba/nnunetv2/evaluation/
cp find_best_configuration.py ./U-Mamba/umamba/nnunetv2/evaluation/