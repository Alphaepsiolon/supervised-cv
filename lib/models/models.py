import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from lib.dataset.dataset import DataPreprocess, ClassificationDataset
from lib.metrics.metrics import compute_metrics
from torch.utils.data import Dataset, DataLoader

class ClassificationModel(pl.LightningModule):
    def __init__(self, num_classes:int = 3, preprocess: DataPreprocess=None, data_transforms:dict=None):
        super().__init__()
        self.preprocess = preprocess
        self.num_classes = num_classes
        self.data_transforms = data_transforms
        self.model = models.resnet18(pretrained=False)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)
        self.sm = nn.Softmax(dim=1)

    def forward(self,x: torch.Tensor):
        x = self.model(x)
        x = self.sm(x)
        return x

    def vanilla_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        dataset = ClassificationDataset(self.preprocess.train_data, transform=self.data_transforms['train_transforms'])
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        return dataloader
    
    def training_step(self, batch):
        x,y = batch

        y_hat = self.forward(x)
        y = F.one_hot(y, num_classes = self.num_classes).float()
        loss = self.vanilla_loss(y_hat, y)

        # compute metrics
        metrics = compute_metrics(y_hat, batch[1])
        self.log('train_loss',loss)
        self.log('accuracy',metrics['accuracy'])
        return loss

    def val_dataloader(self):
        dataset = ClassificationDataset(self.preprocess.val_data, transform=self.data_transforms['val_transforms'])
        dataloader = DataLoader(dataset, batch_size=len(self.preprocess.val_data), shuffle=False, num_workers=4)
        return dataloader

    def validation_step(self, batch, batch_idx):
        x,y = batch

        y_hat = self.forward(x)
        y = F.one_hot(y, num_classes = self.num_classes).float()
        loss = self.vanilla_loss(y_hat, y)
        
        # compute metrics
        metrics = compute_metrics(y_hat, batch[1])
        self.log('val_loss',loss)
        self.log('val_accuracy',metrics['accuracy'])
        return {'loss':loss,
                'y': y,
                'y_hat':y_hat}

    def validation_epoch_end(self, outputs):
        # print(outputs[0]['y'].argmax(-1).flatten().shape)
        # print(outputs[0]['y_hat'].argmax(-1).shape)
        y = outputs[0]['y'].argmax(-1)
        y_hat = outputs[0]['y_hat']
        # print(y)
        # print(y_hat.argmax(-1))
        metrics = compute_metrics(y_hat, y)
        self.log('f1_micro',metrics['f1_score_micro'])
        self.log('f1_macro',metrics['f1_score_macro'])

    