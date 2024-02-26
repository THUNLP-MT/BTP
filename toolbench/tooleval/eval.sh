subset=$1

bash run_convert_answer.sh ours_cost20_th0.15 $subset
bash run_pass_rate.sh ours_cost20_th0.15 $subset
python eval_cost.py ours_cost20_th0.15 $subset
