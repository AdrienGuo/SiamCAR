# bin/bash

# Model Info
train_dataset="all"
criteria="mid"
neg=(0.0)
size=(255)
bg="1.3"
epoch=(1000)
batch=(32)
n_epoch=(1000)

# Test Info
test_dataset="all"


save_models="./save_models/${train_dataset}/${criteria}"
model_dir="${train_dataset}_${criteria}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
n_model="model_e${n_epoch}.pth"
# model path(es)
model="${save_models}/${model_dir}/${n_model}"
# official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"
# dummy_model="./save_models/dummy_model/dummy_model_1.pth"


echo "=== Your Parameters ==="
echo "Model: ${model_dir} & ${n_epoch}th"
echo "Test dataset: ${test_dataset}"
echo "Criteria: ${criteria}"
echo "Size: ${size}"
echo "Background: ${bg}"
sleep 3


python3 \
    ./tools/test.py \
    --model ${model} \
    --dataset_name ${test_dataset} \
    --test_dataset ./datasets/test/${test_dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \
