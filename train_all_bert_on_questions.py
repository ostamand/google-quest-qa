from train_bert_on_questions import main as train

def main():
    params = {
        "epochs1": 10,
        "epochs2": 5, 
        "model_dir": "model",
        "data_dir": "data",
        "fold": 0,
        "log_dir": ".logs",
        "seed": 42, 
        "bs": 8,
        "dp": 0.1, 
        "maxlen": 256,
        "device": "cuda",
        "do_apex": True, 
        "do_wandb": True,
        "do_tb": False,
        "warmup": 0.5,
        "warmdown": 0.5, 
        "clip": 10.0,
        "accumulation_steps": 2, 
        "project": "google-quest-qa",
        "head_ckpt": None
    }
    for i in range(5):
        params['fold'] = i
        train(params)

if __name__ == "__main__":
    pass