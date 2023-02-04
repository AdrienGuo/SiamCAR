# bin/bash

part="test"
dataset="all"


echo "Part: ${part}"
echo "Dataset: ${dataset}"
sleep 3


python3 \
    ./pysot/datasets/analysis/label.py \
    --part ${part} \
    --dataset_name ${dataset} \
    --dataset ./datasets/${part}/${dataset} \
