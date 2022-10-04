import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *

from tqdm import trange

class Trainer(object):
    """
    Trainer class for MNIST dataset classification
    """

    def __init__(self, config, dataloader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        if self.config['cuda']:
            self.model.to('cuda')
        self.criterion = getattr(nn, config["train"]["criterion"])()
        self.optimizer = getattr(optim, config["train"]["optimizer"])(
            self.model.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["l2_coeff"],
        )
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.config['train']['gamma'])

        self.writer = SummaryWriter(f'{config["log_dir"]}/{model.__class__.__name__}')

    def train(self):
        self.model.train()
        val_accuracy = {'top1': 0.0, 'top5': 0.0}
        global_i = 0
        for epoch in range(self.config['train']['epoch']):
            bar = trange(len(self.dataloader.train_dataloader))
            for _ in bar:
                train_data = next(iter(self.dataloader.train_dataloader))
                image = train_data["image"]
                label = train_data["label"]

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                bar.set_description(
                    f"Epoch: {epoch} [Loss: {loss.cpu().detach().numpy():0.6f}] [Top1: {val_accuracy['top1']:0.6f}] [Top5: {val_accuracy['top5']:0.6f}]"
                )
                bar.refresh()

                # Run accuracy on validation set
                if global_i % 50 == 0:
                    val_accuracy = self.validate()

                    self.writer.add_scalar(
                        "loss/Cross Entropy Loss",
                        loss.cpu().detach().numpy(),
                        global_i,
                    )
                    self.writer.add_scalar(
                        "metrics/Top 1 accuracy",
                        val_accuracy['top1'],
                        global_i,
                    )
                    self.writer.add_scalar(
                        "metrics/Top 5 accuracy",
                        val_accuracy["top5"],
                        global_i,
                    )

                global_i += 1

        test_accuracy = self.test()
        print(test_accuracy)
        self.save_model()

    def validate(self):
        self.model.eval()
        validate_data = next(iter(self.dataloader.val_dataloader))
        image = validate_data["image"]
        label = validate_data["label"]

        output = self.model(image)
        metric = top_k_accuracy(output, label)
        return metric

    def test(self):
        self.model.eval()
        test_data = next(iter(self.dataloader.test_dataloader))
        image = test_data["image"]
        label = test_data["label"]

        output = self.model(image)
        metric = top_k_accuracy(output, label)
        return metric

    def save_model(self):
        torch.save(self.model.state_dict(), self.config["save_dir"])