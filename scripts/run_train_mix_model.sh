(cd ..; python3 mix_model.py \
        --bs 16 \
        --epochs 100 \
        --lr 1e-4 \
        --warmup 0.5 \
        --warmdown 0.5 \
        --out_dir outputs/mix_model_2 \
        --sub_type 2 \
        --ckpt_dir outputs/bert_on_all_1  )