method=$1
subset=$2

export RAW_ANSWER_PATH=../../data/answer
export CONVERTED_ANSWER_PATH=../../data/converted_answer

mkdir -p ${CONVERTED_ANSWER_PATH}/${method}

answer_dir=${RAW_ANSWER_PATH}/${method}/${subset}
output_file=${CONVERTED_ANSWER_PATH}/${method}/${subset}.json
    
python convert_to_answer_format.py \
    --answer_dir ${answer_dir} \
    --method DFS \
    --output ${output_file}
