(cd ..; python3 from_kaggle.py \
    --data_dir data \
    --out_dir outputs/keras_qa \
    --model_dir model/bert_en_uncased_L-12_H-768_A-12 \
    --fold 0 \
    --bs 8 \
    --lr 2e-5)