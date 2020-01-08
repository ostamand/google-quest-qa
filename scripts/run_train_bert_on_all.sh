(cd ..; python3 train_bert_on_all.py \
        --do_apex  \
        --do_wandb \
        --bs 4 \
        --fold 4 \
        --seed 1 \
        --accumulation_step 2 \
        --epochs 5 \
        --lr 2e-5 \
        --out_dir outputs/bert_on_all_lm_4 \
        --dp 0. \
        --bert_wd 0.01 \
        --max_len_q_b 150 \
        --model_dir model/bert-base-uncased \
        --warmup 0.5 \
        --warmdown 0.5 )