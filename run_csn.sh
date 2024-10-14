export CUDA_VISIBLE_DEVICES=0

lang=ruby
output_dir=saved_models/CSN/$lang
mkdir -p $output_dir
cat run_csn.sh > ${output_dir}/run_csn.sh

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
    --do_test \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee ${output_dir}/run_csn.log
