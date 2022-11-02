# bin/bash


# Model Info
model_dataset="all"
criteria="above"
neg=(0.0)
size=(255)
bg="1.3"
epoch=(1000)
batch=(32)
n_epoch=(50)

# Test Info
test_dataset="PatternMatch_test"


save_models="./save_models"
model_dir="${model_dataset}_${criteria}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
n_model="model_e${n_epoch}.pth"
model="${save_models}/${model_dir}/${n_model}"
# official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"


echo "=== Your Parameters ==="
echo "Model: ${model}"
echo "Test dataset: ${test_dataset}"
echo "Criteria: ${criteria}"
echo "Size: ${size}"
echo "Background: ${bg}"
sleep 3


python3 \
    ./tools/test.py \
    --model ${model} \
    --dataset_name ${test_dataset} \
    --dataset_path ./datasets/test/${test_dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \
