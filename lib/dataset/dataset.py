import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2 as cv

class DataPreprocess(object):
    def __init__(self,
                 path:str,
                 n_samples_train:float = None):
        self.path = path
        self.data_df = self.preprocess()
        self.n_samples_train=n_samples_train
        # create splits
        self.td,self.vd = self.create_splits()

    def preprocess(self):
        img_paths = glob.glob(f"{self.path}/*")
        data = {
            'img':[],
            'label_str':[],
            'label':[]
        }
        for imp in img_paths:
            data['img'].append(imp)
            if 'COVID' in imp:
                data['label'].append(0)
                data['label_str'].append('COVID')
            elif 'NORMAL' in imp:
                data['label'].append(1)
                data['label_str'].append('NORMAL')
            elif 'PNEUMONIA' in imp:
                data['label'].append(2)
                data['label_str'].append('PNEUMONIA')
            else:
                print(imp)
                raise ValueError
        
        data_df = pd.DataFrame(data)
        return data_df
    
    def create_splits(self):
        shuf_data = self.data_df.sample(frac=1).reset_index(drop=True)

        # split data
        train_data,val_data = train_test_split(shuf_data, test_size=0.2, random_state=1)

        if self.n_samples_train:
            train_final = []
            test_final = []
            for l in train_data.label_str.unique().tolist():
                sub = train_data[train_data['label_str']==l]
                # sub = sub.sample(n=self.n_samples_train)
                sub, sub_test = train_test_split(sub, test_size=len(sub)-20, random_state=1)
                train_final.append(sub)
                test_final.append(sub_test)
            train_data = pd.concat(train_final)
            val_data = pd.concat([val_data, *test_final])
        return train_data,val_data
    
    @property
    def train_data(self):
        return self.td
    @property
    def val_data(self):
        return self.vd

class ClassificationDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transform:transforms.Compose = None,
                 target_transform:transforms.Compose = None) -> None:
        self.data = df
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.fromarray(cv.imread(self.data['img'].iloc[idx]))
        label = self.data['label'].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.transform(label)

        return image, label