import torch
import numpy as np

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_accuracy(output, target):
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    top1 = 0.0
    top5 = 0.0
    for idx, label in enumerate(target):
        class_prob = output[idx]
        top_values = (-class_prob).argsort()[:5]
        if top_values[0] == label:
            top1 += 1.0
        if np.isin(np.array([label]), top_values):
            top5 += 1.0
    top1 = top1 / len(target)
    top5 = top5 / len(target)
    return {'top1': top1, 'top5': top5}