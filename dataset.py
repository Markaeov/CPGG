import os
import numpy as np
from torch.utils.data import Dataset
import json
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

class UKBCMR2DBase(Dataset):
    def __init__(self,
                 txt_file,
                 size=96,
                 ):
        self.json_file = txt_file
        self.data = json.load(open(txt_file, "r"))
        for item in self.data:
            file_name = item[0].split('/')[-1].split('.')[0]
            item[0] = os.path.join(f'./cmr_data/ukb/{file_name}', file_name+'.pt')

        self._length = len(self.data)

        self.scaler = StandardScaler()
        self.train_data = pd.DataFrame(self.data)
        self.scaler.fit(self.train_data.iloc[:, 1].values.tolist())

        self.size = size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item[0])
        metrics = item[1]
        metrics = self.scaler.transform([metrics])[0]
        img = np.array(image).astype(np.uint8)

        # origin CMRs were processed to 0-255
        image = (img / 127.5 - 1.0).astype(np.float32) # normalize to -1, 1
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image, metrics


class UKBCMR2DTrain(UKBCMR2DBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="train_ukbdata.json", **kwargs)

class UKBCMR2DValidation(UKBCMR2DBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="val_ukbdata.json", **kwargs)

class UKBCMR2DTrain_build_cache(UKBCMR2DBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="train_ukbdata.json", **kwargs)
    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item[0])
        metrics = item[1]
        img = np.array(image).astype(np.uint8)

        image = (img / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # print(image.shape)  # (bs,) c, f, h, w
        return image, torch.tensor(metrics), item[0].split('/')[-1].split('.')[0]


class UKBCMR2DValidation_build_cache(UKBCMR2DBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="val_ukbdata.json", **kwargs)
    def __getitem__(self, i):
        item = self.data[i]
        image = torch.load(item[0])
        metrics = item[1]
        img = np.array(image).astype(np.uint8)

        image = (img / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        # print(image.shape)  # (bs,) c, f, h, w
        return image, torch.tensor(metrics), item[0].split('/')[-1].split('.')[0]

class UKBCMR_cachedBase(Dataset):
    def __init__(self,
                 txt_file,
                 cached_path,
                 size=96,
                 ):
        assert cached_path is not None
        self.json_file = txt_file
        self.data = json.load(open(txt_file, "r"))
        for item in self.data:
            file_name = item[0].split('/')[-1].split('.')[0]
            item[0] = os.path.join(cached_path, file_name+'.npz')

        self._length = len(self.data)
        self.scaler = StandardScaler()
        self.train_data = pd.DataFrame(self.data)
        self.scaler.fit(self.train_data.iloc[:, 1].values.tolist())

        self.size = size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]
        item_data = np.load(item[0])
        image_moments = item_data['moments']
        metrics = item[1]
        metrics = self.scaler.transform([metrics])[0]

        # print(image_moments.shape)  # (bs*2,) c, f, h, w
        return image_moments, torch.tensor(metrics).float()

class UKBCMR_cachedTrain(UKBCMR_cachedBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="train_ukbdata.json", **kwargs)

class UKBCMR_cachedValidation(UKBCMR_cachedBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="val_ukbdata.json", **kwargs)



class UKBCMR_MetricsBase(Dataset):
    def __init__(self,
                 txt_file,
                 ):
        self.json_file = txt_file
        self.data_file = json.load(open(txt_file, "r"))
        self.data = []
        for item in self.data_file:
            self.data.append(item[1])
        self.data = np.array(self.data)
        # print(self.data.shape)
        self.scaler = StandardScaler()
        self.scaler.fit(self.data)
        # print(self.scaler.mean_)
        self._length = len(self.data)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        item = self.data[i]
        metrics_normalized = self.scaler.transform([item])
        metrics = torch.from_numpy(metrics_normalized).float().squeeze()
        return metrics, 0.0
    
    def inverse_transform(self, normalized_data):
        return self.scaler.inverse_transform(normalized_data)


class UKBCMR_MetricsTrain(UKBCMR_MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="train_ukbdata.json", **kwargs)


class UKBCMR_MetricsValidation(UKBCMR_MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="val_ukbdata.json", **kwargs)