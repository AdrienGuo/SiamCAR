# bin/bash

# Model Info
train_dataset="all"
train_criteria="big"
neg=(0.0)
size=(255)
bg="1.0"
epoch=(1000)
batch=(1)
n_epoch=(300)

# Test Info
part="test"
test_dataset="all"
criteria="big"


# Model path
save_models="./save_models/${train_dataset}/${train_criteria}"
model_dir="${train_dataset}_${train_criteria}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
n_model="model_e${n_epoch}.pth"
model="${save_models}/${model_dir}/${n_model}"
# official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"
# dummy_model="./save_models/dummy_model_titan/dummy_model.pth"


echo "=== Your Test Parameters ==="
# echo "Model: ${model_dir} & ${n_epoch}th"
echo "Test dataset: ${test_dataset}"
echo "Criteria: ${criteria}"
echo "Size: ${size}"
echo "Background: ${bg}"
sleep 3


python3 \
    ./tools/test.py \
    --model ${model} \
    --dataset_name ${test_dataset} \
    --part ${part} \
    --test_dataset ./datasets/${part}/${test_dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \
