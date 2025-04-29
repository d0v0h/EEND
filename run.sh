#!/bin/bash

# global variables setup
PROJ_DIR=$(pwd)
export PYTHONPATH=`pwd`:$PYTHONPATH
NS=2


# Dataset root
DB_DIR=$PROJ_DIR/DB

# Simulated
simu_train_dir=$DB_DIR/simu/train_all_ns"$NS"_beta8_100000
simu_dev_dir=$DB_DIR/simu/dev_all_ns"$NS"_beta5_500
simu_test_dir=$DB_DIR/simu/test_all_ns"$NS"_beta5_500

# Real
real_train_dir=$DB_DIR/real/callhome1_spk"$NS"
real_dev_dir=$DB_DIR/real/callhome2_spk"$NS"

stage=1

# output directory
EXP_NAME=EEND_EDA
output_dir=$PROJ_DIR/exp/$EXP_NAME

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python eend/train.py \
        --config $PROJ_DIR/examples/train.yaml \
        --train-data-dir $simu_train_dir \
        --valid-data-dir $simu_dev_dir \
        --output-path $output_dir/simu || exit 1
fi

# Inference
if [ $stage -le 2 ]; then
    echo "Start inference"
    python eend/infer.py \
        --config $PROJ_DIR/examples/infer.yaml \
        --infer-data-dir $simu_test_dir \
        --models-path $output_dir/simu/models \
        --rttms-dir $output_dir/simu || exit 1
fi

# Scoring
if [ $stage -le 3 ]; then
    echo "Start scoring"
    ./md-eval.pl \
        -c 0.25 \
        -r $simu_test_dir/rttm \
        -s $output_dir/simu/rttms/ref_0.5.rttm \
        > $output_dir/simu/result_th0.5_med11_collar0.25 2>/dev/null || exit 1
fi

# Adapting
if [ $stage -le 4 ]; then
    echo "Start adapting"
    python eend/train.py \
        --config $PROJ_DIR/examples/adapt.yaml \
        --init-model-path $output_dir/simu/models \
        --train-data-dir $real_train_dir \
        --valid-data-dir $real_dev_dir \
        --output-path $output_dir/real || exit 1
fi

# Inference
if [ $stage -le 5 ]; then
    echo "Start inference"
    python eend/infer.py \
        --config $PROJ_DIR/examples/infer.yaml \
        --infer-data-dir $real_dev_dir \
        --models-path $output_dir/real/models \
        --rttms-dir $output_dir/real || exit 1
fi

# Scoring
if [ $stage -le 6 ]; then
    echo "Start scoring"
    ./md-eval.pl \
        -c 0.25 \
        -r $real_dev_dir/rttm \
        -s $output_dir/real/rttms/ref_0.5.rttm \
        > $output_dir/real/result_th0.5_med11_collar0.25 2>/dev/null || exit 1
fi

echo "Done"