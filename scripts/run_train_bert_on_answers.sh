(cd ..; python3 train_bert_on.py \
        --do_apex \
        --do_wandb \
        --maxlen 512 \
        --bs 4 \
        --accumulation_steps 2 \
        --dp 0.2 \
        --lr1 1e-3 \
        --fold 0 \
        --do_answer \
        --out_dir outputs/bert_on_answers)
# will not do head