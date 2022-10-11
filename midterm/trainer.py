import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from utils.utils import WarmUpLR
from tqdm import trange


class Trainer(object):
    """
    Trainer class for CIFAR dataset classification
    """

    def __init__(self, config, dataloader, model):
        self.config = config
        self.dataloader = dataloader
        self.model = model
        if self.config["cuda"]:
            self.model.to("cuda")
        self.criterion = getattr(nn, config["train"]["criterion"])()
        self.optimizer = getattr(optim, config["train"]["optimizer"])(
            self.model.parameters(),
            lr=config["train"]["learning_rate"],
            momentum=config["train"]["momentum"],
            weight_decay=config["train"]["weight_decay"],
        )
        self.scheduler = getattr(
            torch.optim.lr_scheduler, config["train"]["scheduler"]
        )(
            self.optimizer,
            milestones=self.config["train"]["milestones"],
            gamma=self.config["train"]["gamma"],
        )
        self.warmup_scheduler = WarmUpLR(
            self.optimizer,
            len(self.dataloader.train_dataloader) * self.config["train"]["warm"],
        )

        self.writer = SummaryWriter(f'{config["log_dir"]}/{model.__class__.__name__}')

    def train(self):
        start = time.time()
        self.model.train()
        test_result = {"top1": 0.0, "top5": 0.0}
        global_i = 0
        for epoch in range(self.config["train"]["epoch"]):

            if epoch >= 1.0:
                self.scheduler.step()

            for idx, (image, label) in enumerate(self.dataloader.train_dataloader):
                if self.config["cuda"]:
                    image = image.to("cuda")
                    label = label.to("cuda")

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                print(
                    f"Epoch: {epoch} Iter: {idx}/{len(self.dataloader.train_dataloader)} [Loss: {loss.cpu().detach().numpy():0.6f}] [Top1: {test_result['top1']:0.6f}] [Top5: {test_result['top5']:0.6f}] LR:{self.optimizer.param_groups[0]['lr']}"
                )

                if global_i % 20 == 0:
                    self.writer.add_scalar(
                        "train/Cross Entropy Loss",
                        loss.cpu().detach().numpy(),
                        global_i,
                    )

                if epoch < self.config["train"]["warm"]:
                    self.warmup_scheduler.step()

                global_i += 1

            # Run accuracy on test set
            test_result = self.test()

            self.writer.add_scalar(
                "test/Top 1 accuracy",
                test_result["top1"],
                epoch,
            )
            self.writer.add_scalar(
                "test/Top 5 accuracy",
                test_result["top5"],
                epoch,
            )

            print(f'Time elapsed for epoch {epoch}: {time.time() - start}')
            start = time.time()

        self.save_model()

    def test(self):
        self.model.eval()

        top1 = 0.0
        top5 = 0.0
        for image, label in self.dataloader.test_dataloader:
            if self.config["cuda"]:
                image = image.to("cuda")
                label = label.to("cuda")

            output = self.model(image)

            output = output.cpu().detach().numpy()
            target = label.cpu().detach().numpy()

            for idx, label in enumerate(target):
                class_prob = output[idx]
                top_values = (-class_prob).argsort()[:5]
                if top_values[0] == label:
                    top1 += 1.0
                if np.isin(np.array([label]), top_values):
                    top5 += 1.0

        top1 = top1 / len(self.dataloader.test_dataloader.dataset)
        top5 = top5 / len(self.dataloader.test_dataloader.dataset)

        return {"top1": top1, "top5": top5}

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            f'{self.config["save_dir"]}/{self.model.__class__.__name__}.pth',
        )
