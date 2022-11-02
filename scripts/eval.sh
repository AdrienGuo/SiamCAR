# bin/bash

dataset="all"
criteria="above"
neg=(0.0)
size=(255)
bg="1.3"
epoch=(1000)
batch=(32)
n_epoch=(1000)


save_models="./save_models"
model_dir="allOld_${criteria}_neg${neg}_x${size}_bg${bg}_e${epoch}_b${batch}"
n_model="model_e${n_epoch}.pth"
model="${save_models}/${model_dir}/${n_model}"
# official_model="./snapshot/official/model_general.pth"
# amy_model="./snapshot/amy/checkpoint_e999.pth"
# other_model="./snapshot/others/checkpoint_e199.pth"


echo "=== Your Parameters: ==="
echo "Model: ${model_dir}"
echo "Test dataset: ${dataset}"
echo "Criteria: ${criteria}"
echo "Size: ${size}"
echo "Background: ${bg}"
sleep 3


python3 \
    ./tools/eval.py \
    --model ${model} \
    --dataset_name ${dataset} \
    --test_dataset ./datasets/test/${dataset} \
    --criteria ${criteria} \
    --neg ${neg} \
    --bg ${bg} \