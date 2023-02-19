# bin/bash

# Colors for terminal
RED="\e[31m"
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Model Info
loss_method="bce"  # bce / focal
train_date="02042023"
train_dataset="all"  # all
train_criteria="all"  # all
train_target="multi"  # multi
train_method="siamcar"  # siamcar / origin / search
neg=(0.0)  # 負樣本的比率
size=(255)
bg="1.0"  # 使用多少 background
epoch=(200)
batch=(1)
ckpt=(6)
cfg_name="config"
cfg="./experiments/${train_date}/${cfg_name}.yaml"

# Demo Info
dataset="test"  # train / test
part="amy"  # all / tmp
criteria="mid"  # all / big / mid / small
target="multi"  # one / multi
method="origin"  # origin / tri_origin / search

# Model path
model_dir="./models/${train_date}/${cfg_name}/${train_dataset}/${train_criteria}/${train_target}/${train_method}"
model_name="${train_dataset}_${train_criteria}_${train_target}_${train_method}"
ckpt="ckpt${ckpt}"
my_model="${model_dir}/${model_name}/${ckpt}.pth"
official_model="./pretrained_models/model_general.pth"
amy_model="./snapshot/amy/checkpoint_e999.pth"
other_model="./snapshot/others/checkpoint_e199.pth"
dummy_model="./models/dummy_model_titan/dummy_model_1.pth"
# Model to use
model=${official_model}

echo "=== Your Evaluate Parameters ==="
echo -e "Model: ${RED}${model} ${ENDCOLOR}"
echo -e "Dataset: ${GREEN}${dataset} ${ENDCOLOR}"
echo -e "Part: ${GREEN}${part} ${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria} ${ENDCOLOR}"
echo -e "Target: ${GREEN}${target} ${ENDCOLOR}"
echo -e "Method: ${GREEN}${method} ${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg} ${ENDCOLOR}"
echo -e "Config: ${GREEN}${cfg} ${ENDCOLOR}"
sleep 3

python3 \
    ./tools/evaluate.py \
    --model ${model} \
    --data ./data/TRI/${dataset}/${part} \
    --dataset ${dataset} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --neg ${neg} \
    --bg ${bg} \
    --cfg ${cfg}
