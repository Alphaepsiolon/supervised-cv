import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms,models
from lib.dataset.dataset import DataPreprocess
from lib.models.models import ClassificationModel

if __name__ == '__main__':
    preprocess = DataPreprocess('/home/ubuntu/adithya/temp/EAMLA/Computer Vision/diseases', n_samples_train=20)
    data_transforms = {
        'train_transforms': transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        'val_transforms': transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
        'label_transforms': transforms.Compose([
                    transforms.ToTensor()
        ])
    }

    model = ClassificationModel(3,preprocess=preprocess, data_transforms=data_transforms)
    wandb_logger = WandbLogger(project='vision-xray', name = "20 Train Baseline no pre")

    checkpoint_callback = ModelCheckpoint(dirpath="/home/ubuntu/adithya/temp/EAMLA/Computer Vision/logs20", save_top_k=5, monitor="val_loss", filename='xray_classifier-{epoch:02d}-{val_loss:.2f}')
    trainer= pl.Trainer(max_epochs = 50, logger=wandb_logger, callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10), checkpoint_callback])
    trainer.fit(model)