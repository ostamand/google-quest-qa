(cd ..; python3 train_bert_on_answers.py \
        --do_apex \
        --do_wandb \
        --bs 4 \
        --accumulation_steps 2 \
        --lr2 2e-5 \
        --fold 4 \
        --model_dir outputs/lm_finetuning_all \
        --out_dir outputs/bert_on_answers_1 )