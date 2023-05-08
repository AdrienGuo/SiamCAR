# bin/bash

part="train"
dataset="_corrected_labels"

echo "Part: ${part}"
echo "Dataset: ${dataset}"
sleep 3


python3 \
    ./pysot/datasets/analysis/label.py \
    --part ${part} \
    --dataset_name ${dataset} \
    --dataset ./data/TRI/${part}/${dataset} \
