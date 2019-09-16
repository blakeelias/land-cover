#!/bin/bash

LOSSES=(
    "crossentropy"
    "jaccard"
    "superres"
)

MODEL_TYPES=(
    "unet"
    "unet_large"
    "fcdensenet"
)

STATES=(
    north,
    south
)

GPU_ID=0
LOSS=${LOSSES[0]}
MODEL_TYPE=${MODEL_TYPES[0]}

((TIME_BUDGET=3600*12))
BATCH_SIZE=16
LEARNING_RATE=0.001

TRAIN_STATE_LIST="south"
VAL_STATE_LIST="north"
SUPERRES_STATE_LIST="ny_1m_2013"


EXP_NAME=CVPR-for_github-loss-${LOSS}-model-${MODEL_TYPE}-training_states-${TRAIN_STATE_LIST// /-}
OUTPUT=/mnt/blobfuse/train-output/ICLR_2020

if [ -d "${OUTPUT}/${EXP_NAME}" ]; then
    echo "Experiment ${OUTPUT}/${EXP_NAME} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${OUTPUT}/${EXP_NAME}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${OUTPUT}/${EXP_NAME}/
cp -r *.sh *.py ${OUTPUT}/${EXP_NAME}/


LOG_FILE=${OUTPUT}/${EXP_NAME}/log.txt

echo ${LOG_FILE}

# unbuffer python -u train_model_landcover.py \
#     --output ${OUTPUT} \
#     --name ${EXP_NAME} \
#     --gpu ${GPU_ID} \
#     --verbose 1 \
#     --data_dir /home/caleb/data/ \
#     --training_states ${TRAIN_STATE_LIST} \
#     --validation_states ${VAL_STATE_LIST} \
#     --superres_states ${SUPERRES_STATE_LIST} \
#     --model_type ${MODEL_TYPE} \
#     --learning_rate ${LEARNING_RATE} \
#     --loss ${LOSS} \
#     --batch_size ${BATCH_SIZE} \
#     --time_budget ${TIME_BUDGET} \
#     &> ${LOG_FILE} &

# tail -f ${LOG_FILE}

python -u train_model_landcover.py \
    --output ${OUTPUT} \
    --name ${EXP_NAME} \
    --gpu ${GPU_ID} \
    --verbose 1 \
    --data_dir /mnt/blobfuse/cnn-minibatches/clustering/ \
    --training_states ${TRAIN_STATE_LIST} \
    --validation_states ${VAL_STATE_LIST} \
    --superres_states ${SUPERRES_STATE_LIST} \
    --model_type ${MODEL_TYPE} \
    --learning_rate ${LEARNING_RATE} \
    --loss ${LOSS} \
    --batch_size ${BATCH_SIZE} \
    --time_budget ${TIME_BUDGET}

#wait;
exit
