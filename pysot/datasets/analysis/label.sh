# bin/bash

part="train"
dataset="wrong_labels"
criteria="all"


echo "Part: ${part}"
echo "Dataset: ${dataset}"
echo "Criteria: ${criteria}"
sleep 3


python3 \
    ./pysot/datasets/analysis/label.py \
    --part ${part} \
    --dataset_name ${dataset} \
    --dataset ./datasets/${part}/${dataset} \
    --criteria ${criteria} \
