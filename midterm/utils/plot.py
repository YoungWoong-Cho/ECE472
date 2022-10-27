import matplotlib.pyplot as plt

# B, S, Ti
data = {
    "ViT": {
        "GFLOPS": [4.326590976, 4.326590976, 4.326590976],
        "train_loss": [],
        "train_accuracy": [0.39452, 0.39894],
        "eval_accuracy": [0.3253, 0.3327]
    },
    "PiT": {
        "GFLOPS": [2.761954304, 0.701461728, 0.150245248],
        "train_loss": [],
        "train_accuracy": [0.54568, 0.4706],
        "eval_accuracy": [0.4244, 0.3981]
    }
}

def plot_model_capability():
    pass