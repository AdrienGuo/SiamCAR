# bin/bash

dataset="all"
criteria="mid"
neg=(0.0)
bg="1.3"


echo "=== Your Training Parameters ==="
echo "Dataset: ${dataset}"
echo "Criteria: ${criteria}"
echo "Neg Ratio: ${neg}"
echo "Background: ${bg}"
echo "- Check your NECK crop"
echo "- Check where PCBDataset import from"
echo "- Check your Learning Rate"
sleep 3


python3 \
    ./pysot/datasets/analysis/analysis.py \
    --dataset_name ${dataset} \
    --dataset ./datasets/train/${dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \
