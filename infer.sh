export TOOLBENCH_KEY=$YOUR_TOOLBENCH_KEY
export OPENAI_KEY=$YOUR_OPENAI_KEY
export PYTHONPATH=./

METHOD=ours_cost20_th0.15
SUBSET=$1

mkdir -p data/answer/$METHOD/$SUBSET

python -u toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model chatgpt_function \
    --openai_key $OPENAI_KEY \
    --max_observation_length 1024 \
    --method DFS_woFilter_w2 \
    --input_query_file inst/$METHOD/$SUBSET.json \
    --output_answer_file data/answer/$METHOD/$SUBSET \
    --toolbench_key $TOOLBENCH_KEY \
    --use_blacklist
