import numpy as np

from config import CONFIG
from tqdm import tqdm

def top_k_accuracy(model, dataloader):
    model.eval()
    top1 = 0.0
    top5 = 0.0
    
    for images, labels in tqdm(dataloader):
        images = images.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        output = model(images)

        output = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for idx, label in enumerate(labels):
            class_prob = output[idx]
            top_values = (-class_prob).argsort()[:5]
            if top_values[0] == label:
                top1 += 1.0
            if np.isin(np.array([label]), top_values):
                top5 += 1.0

    top1 = top1 / len(dataloader.dataset)
    top5 = top5 / len(dataloader.dataset)
    return {"top1": top1, "top5": top5}
