export CUDA_VISIBLE_DEVICES=0

output_dir=saved_models/AdvTest
mkdir -p $output_dir
cat run_advtest.sh > ${output_dir}/run_advtest.sh

python run.py \
    --output_dir ${output_dir} \
    --model_name_or_path microsoft/unixcoder-base \
    --temp 0.05 \
    --queue_size 8192 \
    --momentum 0.999 \
    --alpha 0.4 \
    --no_mome \
    --distill \
    --do_train \
    --train_data_file dataset/AdvTest/train.jsonl \
    --eval_data_file dataset/AdvTest/valid.jsonl \
    --codebase_file dataset/AdvTest/valid.jsonl \
    --num_train_epochs 5 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee ${output_dir}/run_advtest.log
