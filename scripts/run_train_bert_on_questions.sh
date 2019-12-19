(cd ..; python3 train_bert_on.py \
        --do_apex \
        --do_wandb \
        --maxlen 512 \
        --bs 4 \
        --accumulation_steps 2 \
        --dp 0.2 \
        --lr1 1e-2 \
        --fold 0 \
        --do_head \
        --out_dir outputs/bert_on_questions)
# TODO:
# - Try dp 0.2