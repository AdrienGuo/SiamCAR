# bin/bash

dataset="all"
criteria="all"
neg=(0.0)
bg="1.3"
epoch=(1000)
batch_size=(32)
accum_iter=(1)


echo "=== Your Training Parameters === "
echo "Dataset: ${dataset}"
echo "Criteria: ${criteria}"
echo "Neg Ratio: ${neg}"
echo "Background: ${bg}"
echo "Epoch: ${epoch}"
echo "Batch: ${batch_size}"
echo "Accum iter: ${accum_iter}"
echo "- Check your NECK crop"
echo "- Check where PCBDataset import from"
echo "- Check your Learning Rate"
sleep 3


python3 \
    ./tools/train.py \
    --dataset_name ${dataset} \
    --dataset ./datasets/train/${dataset} \
    --test_dataset ./datasets/test/${dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --accum_iter ${accum_iter}
