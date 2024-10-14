export CUDA_VISIBLE_DEVICES=0

output_dir=saved_models/CosQA
mkdir -p $output_dir
cat run_cosqa.sh > ${output_dir}/run_cosqa.sh

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
    --train_data_file dataset/cosqa/cosqa-retrieval-train-19604.json \
    --eval_data_file dataset/cosqa/cosqa-retrieval-dev-500.json \
    --test_data_file dataset/cosqa/cosqa-retrieval-test-500.json \
    --codebase_file dataset/cosqa/code_idx_map.txt \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee ${output_dir}/run_cosqa.log
