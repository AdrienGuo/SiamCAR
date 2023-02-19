# bin/bash

# Colors for terminal
RED="\e[31m"
GREEN="\e[32m"
ENDCOLOR="\e[0m"

# Train Settings
date="02092023"
dataset="amy"  # all / tmp
criteria="all"  # all / mid
method="siamcar"  # siamcar / origin / official_origin / siamfc
bg="All"  # background
# Test Settings
test_dataset="all"  # all / tmp
# Evaluate Settings
eval_criteria="all"  # all / mid
eval_method="origin"  # siamcar / origin / official_origin / siamfc
eval_bg="1.0"
# Others Settings
target="multi"  # one / multi
cfg_name="config"
cfg="./experiments/${date}/${cfg_name}.yaml"

# Double Check
echo -e "${GREEN}=== Your Train Parameters ===${ENDCOLOR}"
echo -e "Train Dataset: ${GREEN}${dataset}${ENDCOLOR}"
echo -e "Train Criteria: ${GREEN}${criteria}${ENDCOLOR}"
echo -e "Train Method: ${GREEN}${method}${ENDCOLOR}"
echo -e "Background: ${GREEN}${bg}${ENDCOLOR}"
echo -e "Test Dataset: ${GREEN}${test_dataset}${ENDCOLOR}"
echo -e "Eval Criteria: ${GREEN}${eval_criteria}${ENDCOLOR}"
echo -e "Eval Method: ${GREEN}${eval_method}${ENDCOLOR}"
echo -e "Eval Background: ${GREEN}${eval_bg}${ENDCOLOR}"
echo -e "Target: ${GREEN}${target}${ENDCOLOR}"
echo -e "${RED}Check your config setting!! ${ENDCOLOR}"
echo -e "Config: ${GREEN}${cfg} ${ENDCOLOR}"
sleep 1

# python3 script
python3 \
    tools/train.py \
    --date ${date} \
    --dataset ${dataset} \
    --data "./data/TRI/test/${dataset}" \
    --criteria ${criteria} \
    --method ${method} \
    --bg ${bg} \
    --test_data "./data/TRI/test/${test_dataset}" \
    --eval_criteria ${eval_criteria} \
    --eval_method ${eval_method} \
    --eval_bg ${eval_bg} \
    --target ${target} \
    --cfg_name ${cfg_name} \
    --cfg ${cfg}
