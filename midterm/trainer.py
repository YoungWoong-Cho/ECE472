import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from utils.helpers import WarmUpLR
from utils.metrics import top_k_accuracy


class Trainer(object):
    """
    Trainer class for CIFAR dataset classification
    """

    def __init__(self, dataloader, model):
        self.dataloader = dataloader
        self.model = model
        self.model.to(CONFIG['device'])
        self.criterion = getattr(nn, CONFIG["train"]["criterion"])()
        self.optimizer = getattr(optim, CONFIG["train"]["optimizer"])(
            self.model.parameters(),
            lr=CONFIG["train"]["learning_rate"],
            momentum=CONFIG["train"]["momentum"],
            weight_decay=CONFIG["train"]["weight_decay"],
        )
        self.scheduler = getattr(
            torch.optim.lr_scheduler, CONFIG["train"]["scheduler"]
        )(
            self.optimizer,
            T_max=CONFIG['train']['epoch'],
            eta_min=0.0,
            last_epoch= -1,
            verbose=False
        )
        self.warmup_scheduler = WarmUpLR(
            self.optimizer,
            len(self.dataloader.train_dataloader) * CONFIG["train"]["warm"],
        )

        self.writer = SummaryWriter(f'{CONFIG["log_dir"]}/{model.__class__.__name__}')

    def train(self):
        self.model.train()
        train_accuracy = 0.0
        val_accuracy = 0.0
        global_i = 0
        for epoch in range(CONFIG["train"]["epoch"]):

            if epoch >= 1.0:
                self.scheduler.step()

            for idx, (image, label) in enumerate(self.dataloader.train_dataloader):
                image = image.to(CONFIG['device'])
                label = label.to(CONFIG['device'])

                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, label)
                if not loss.requires_grad:
                    loss.requires_grad = True
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG['train']['grad_clip'])
                self.optimizer.step()

                print(
                    f"Epoch: {epoch} Iter: {idx}/{len(self.dataloader.train_dataloader)} [Train loss: {loss.cpu().detach().numpy():0.6f}]", end=' '
                )
                print(
                    f"[Val acc: {val_accuracy}]"
                )

                if global_i % CONFIG['train']['log_iter'] == 0:
                    self.writer.add_scalar(
                        "train/Cross Entropy Loss",
                        loss.cpu().detach().numpy(),
                        global_i,
                    )
                    self.writer.add_scalar(
                        "train/LR",
                        self.optimizer.param_groups[0]['lr'],
                        global_i,
                    )

                if epoch < CONFIG["train"]["warm"]:
                    self.warmup_scheduler.step()

                global_i += 1

            # Run accuracy
            val_accuracy = top_k_accuracy(self.model, self.dataloader.test_dataloader)
            self.writer.add_scalar(
                "test/Top 1 accuracy",
                val_accuracy['top1'],
                epoch,
            )
            self.writer.add_scalar(
                "test/Top 5 accuracy",
                val_accuracy['top5'],
                epoch,
            )

        train_accuracy = top_k_accuracy(self.model, self.dataloader.train_dataloader)
        val_accuracy = top_k_accuracy(self.model, self.dataloader.test_dataloader)
        print(f'[Train acc: {train_accuracy}] [Val acc: {val_accuracy}]')
        self.save_model()

    def save_model(self):
        torch.save(
            self.model.state_dict(),
            f'{CONFIG["save_dir"]}/{self.model.__class__.__name__}.pth',
        )
