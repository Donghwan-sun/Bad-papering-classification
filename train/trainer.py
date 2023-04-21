import os
import torch

from common.config import settings

class Trainner:
    def __init__(self, train_dataloader, validation_loader, device, optimizer, model, loss_fn, schedule=None):
        self.epoch = settings.EPOCHS
        self.train_dataloader = train_dataloader
        self.test_dataloader = validation_loader
        self.device = device
        self.optimizer = optimizer
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.best_acc = 0
        self.scheduler = schedule

    def train_step(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for datas in self.train_dataloader:
            input = datas["image"].to(self.device)
            target = datas["target"].type(torch.LongTensor).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input)
            loss = self.loss_fn(outputs, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * input.size(0)
            correct += (outputs.argmax(dim=1) == target).float().mean()

        accuracy = correct / len(self.train_dataloader)
        avg_train_loss = total_loss / len(self.train_dataloader)

        return accuracy, avg_train_loss

    def validation_step(self):
        self.model.eval()
        total_val_loss = 0
        correct = 0
        with torch.no_grad():
            for datas in self.test_dataloader:
                input = datas["image"].to(self.device)
                target = datas["target"].type(torch.LongTensor).to(self.device)
                outputs = self.model(input)
                _, preds = torch.max(input, 1)
                loss = self.loss_fn(outputs, target)
                total_val_loss += loss.item() * input.size(0)
                correct += (outputs.argmax(dim=1) == target).float().mean()

        accuracy = correct / len(self.train_dataloader)
        avg_var_loss = total_val_loss / len(self.train_dataloader)

        return accuracy, avg_var_loss

    def fit(self):
        for epoch in range(settings.EPOCHS):
            train_accuracy, avg_train_loss = self.train_step()
            val_accuracy, avg_val_loss = self.validation_step()

            log_msg = (
                f"Epoch [{epoch + 1}/{settings.EPOCHS}] "
                f"Training Loss: {avg_train_loss:.4f} "
                f"Training Accuracy: {train_accuracy:.4f} "
                f"Validation Loss: {avg_val_loss:.4f} "
                f"Validation Accuracy: {val_accuracy:.4f} "
            )
            print(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()

            if self.best_acc < val_accuracy:
                self.best_acc = val_accuracy
                best_model = self.model
                torch.save(best_model, os.path.join(settings.MODEL_PATH,
                                                    f"{settings.PRETRAINED_VERSION}"
                                                    f"-{epoch}"
                                                    f"{settings.MODEL_EXTENSION}"))


