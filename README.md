# Google QUEST Q&A Labeling

## Setup

Port forward on local machine for jupyter notebook.

```
ssh -N -L 8080:localhost:8080 home
```

### Keras Mixed Precision 

Reference: [link](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)

```python
opt = tf.keras.optimizers.Adam()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
model.compile(loss=loss, optimizer=opt)
model.fit(...)
```

## Reference

- https://www.kaggle.com/abhishek/distilbert-use-features-oof
- https://www.kaggle.com/ldm314/quest-encoding-ensemble

## Misc

Check GPU usage

```
nvidia-smi -l 1
```

Run command on existing docker container

```
docker exec -it 10fe76a885c9 bash
```
