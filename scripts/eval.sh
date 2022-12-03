# bin/bash

# Model Info
loss_method="bce"
train_dataset="all"
train_criteria="mid"
train_target="one"
train_method="origin"
neg=(0.0)
size=(255)
bg="1.0"
epoch=(1000)
batch=(1)
ckpt=(220)

# Eval Info
part="train"
test_dataset="all"
criteria="mid"
target="one"
method="origin"


# Model path
save_models="./save_models/${train_dataset}/${train_criteria}/${train_target}/${train_method}"
# model_dir="${loss_method}_${train_dataset}_${train_criteria}_${train_target}_${train_method}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
model_dir="bce_all_mid_one_origin_1.0_1.0_6.0_neg0.0_x255_bg1.0_e1000_b1"
ckpt="ckpt${ckpt}"
model="${save_models}/${model_dir}/${ckpt}.pth"
# official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"
# dummy_model="./save_models/dummy_model_titan/dummy_model_1.pth"


echo "=== Your Eval Parameters ==="
echo "Model: ${model_dir} & ${ckpt}"
echo "Part: ${part}"
echo "Test Dataset: ${test_dataset}"
echo "Criteria: ${criteria}"
echo "Target: ${target}"
echo "Method: ${method}"
echo "Background: ${bg}"
sleep 3


python3 \
    ./tools/eval.py \
    --model ${model} \
    --dataset_name ${test_dataset} \
    --test_dataset ./datasets/${part}/${test_dataset} \
    --criteria ${criteria} \
    --target ${target} \
    --method ${method} \
    --neg ${neg} \
    --bg ${bg} \
