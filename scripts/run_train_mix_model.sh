(cd ..; python3 mix_model.py \
        --do_cache \
        --bs 32 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup 0.5 \
        --warmdown 0.5 \
        --seed 42 \
        --sub_type 1 \
        --model_dir outputs/lm_finetuning_all \
        --ckpt_questions_dir outputs/bert_on_questions_1 \
        --ckpt_mixed_dir outputs/mix_model_3 \
        --out_dir outputs/mix_model_3 \
        --ckpt_dir outputs/bert_on_all_lm_2 )