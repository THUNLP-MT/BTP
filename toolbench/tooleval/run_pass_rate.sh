method=$1
subset=$2

export CONVERTED_ANSWER_PATH=../../data/converted_answer
export SAVE_PATH=pass_rate_results
export API_POOL_FILE=$YOUR_API_POOL_FILE

mkdir -p $SAVE_PATH

python -u eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --subset $subset \
    --save_path ${SAVE_PATH} \
    --reference_model $method \
    --test_ids ../../data/test_query_ids/ \
    --max_eval_threads 1 \
    --evaluate_times 1
