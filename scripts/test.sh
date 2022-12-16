# bin/bash

# Color for terminal
GREEN="\e[32m"
ENDCOLOR="\e[0m"


# Model Info
loss_method="bce"  # bce / focal
train_dataset="all"  # all
train_criteria="mid"  # big / mid / small
train_target="one"  # one / multi
train_method="origin"  # origin / search / official
neg=(0.0)  # 負樣本的比率
size=(255)
bg="1.0"  # 使用多少 background
epoch=(1000)
batch=(1)
ckpt=(1)

# Test Info
part="test"  # train / test
test_dataset="PatternMatch_test"  # all / PatternMatch_test
criteria="all"  # all / big / mid / small
target="multi"  # one / multi
method="tri_origin"  # origin / tri_origin / search


# Model path
save_models="./save_models/${train_dataset}/${train_criteria}/${train_target}/${train_method}"
# model_dir="${loss_method}_${train_dataset}_${train_criteria}_${train_target}_${train_method}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
model_dir="bce_all_mid_one_origin_1.0_1.0_6.0_neg0.0_x255_bg1.0_e1000_b1"
ckpt="ckpt${ckpt}"
# model="${save_models}/${model_dir}/${ckpt}.pth"
official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"
# dummy_model="./save_models/dummy_model_titan/dummy_model_1.pth"

echo "=== Your Test Parameters ==="
# echo "Model: ${model_dir} & ${ckpt}"
echo -e "Part: ${GREEN}${part} ${ENDCOLOR}"
echo -e "Test Dataset: ${GREEN}${test_dataset} ${ENDCOLOR}"
echo -e "Criteria: ${GREEN}${criteria} ${ENDCOLOR}"
echo -e "Target: ${GREEN}${target} ${ENDCOLOR}"
echo -e "Method: ${GREEN}${method} ${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg} ${ENDCOLOR}"
sleep 3


python3 \
    ./tools/test.py \
    --model ${official_model} \
    --dataset_name ${test_dataset} \
    --part ${part} \
    --test_dataset ./datasets/${part}/${test_dataset} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --neg ${neg} \
    --bg ${bg} \
