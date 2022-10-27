import matplotlib.pyplot as plt

# B, S, Ti
data = {
    "ViT": {
        "GFLOPS": [4.326590976, 1.809307008, 0.816179904],
        "train_loss": [2.310760, 2.278235, 2.299069],
        "train_accuracy": [0.39452, 0.39894, 0.39734],
        "eval_accuracy": [0.3253, 0.3327, 0.3318]
    },
    "PiT": {
        "GFLOPS": [2.761954304, 0.701461728, 0.150245248],
        "train_loss": [1.657457, 1.916154, 2.196895],
        "train_accuracy": [0.54568, 0.4706, 0.43668],
        "eval_accuracy": [0.4244, 0.3981, 0.3796]
    }
}

def plot_model_capability():
    vit_x = data['ViT']['GFLOPS']
    vit_y = data['ViT']['train_loss']
    pit_x = data['PiT']['GFLOPS']
    pit_y = data['PiT']['train_loss']

    plt.plot(vit_x, vit_y, 'r')
    plt.plot(vit_x, vit_y, 'ro', label='ViT')
    plt.plot(pit_x, pit_y, 'b')
    plt.plot(pit_x, pit_y, 'bo', label='PiT')

    plt.xlabel('GFLOPs')
    plt.ylabel('Train loss')

    plt.legend()
    plt.grid()
    # plt.show()

def plot_generalization_performance():
    vit_x = [d * 100 for d in data['ViT']['train_accuracy']]
    vit_y = [d * 100 for d in data['ViT']['eval_accuracy']]
    pit_x = [d * 100 for d in data['PiT']['train_accuracy']]
    pit_y = [d * 100 for d in data['PiT']['eval_accuracy']]

    plt.plot(vit_x, vit_y, 'r')
    plt.plot(vit_x, vit_y, 'ro', label='ViT')
    plt.plot(pit_x, pit_y, 'b')
    plt.plot(pit_x, pit_y, 'bo', label='PiT')

    plt.xlabel('Training accuracy')
    plt.ylabel('Validation accuracy')

    plt.legend()
    plt.grid()
    # plt.show()

def plot_model_performance():
    vit_x = data['ViT']['GFLOPS']
    vit_y = [d * 100 for d in data['ViT']['eval_accuracy']]
    pit_x = data['PiT']['GFLOPS']
    pit_y = [d * 100 for d in data['PiT']['eval_accuracy']]

    plt.plot(vit_x, vit_y, 'r')
    plt.plot(vit_x, vit_y, 'ro', label='ViT')
    plt.plot(pit_x, pit_y, 'b')
    plt.plot(pit_x, pit_y, 'bo', label='PiT')

    plt.xlabel('GFLOPs')
    plt.ylabel('Validation accuracy')

    plt.legend()
    plt.grid()
    # plt.show()

if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (20, 4)

    plt.subplot(1, 3, 1)
    plot_model_capability()

    plt.subplot(1, 3, 2)
    plot_generalization_performance()
    
    plt.subplot(1, 3, 3)
    plot_model_performance()

    plt.show()