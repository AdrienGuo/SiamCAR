# bin/bash

# Color for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"


dataset="all"
criteria="all"  # big / mid / small
target="multi"  # one / multi
method="origin"  # origin / search
neg=(0.0)
bg="1.0"  # 使用多少 background
epoch=(200)
batch_size=(1)
accum_iters=(1)


echo -e "${GREEN}=== Your Training Parameters ===${ENDCOLOR}"
echo -e "Dataset: ${GREEN}${dataset}${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
echo -e "Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Neg Ratio: ${GREEN}${neg}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
echo -e "Epoch: ${GREEN}${epoch}${ENDCOLOR}"
echo -e "Batch: ${GREEN}${batch_size}${ENDCOLOR}"
echo -e "Accum Iters: ${GREEN}${accum_iters}${ENDCOLOR}"
echo -e "${GREEN}- Check your NECK crop${ENDCOLOR}"
echo -e "${GREEN}- Check your Learning Rate${ENDCOLOR}"
echo -e "${GREEN}- Check where PCBDataset import from${ENDCOLOR}"
sleep 3


python3 \
    ./tools/train.py \
    --dataset_name ${dataset} \
    --dataset ./datasets/train/${dataset} \
    --test_dataset ./datasets/test/${dataset} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --neg ${neg} \
    --bg ${bg} \
    --epoch ${epoch} \
    --batch_size ${batch_size} \
    --accum_iters ${accum_iters}
