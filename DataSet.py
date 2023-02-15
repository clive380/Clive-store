import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


num_steps=23
features = ['  exx', '  eyy']

# datas = pd.read_csv("数据集/4/1-00001_0.csv", dtype=float)
# datas.columns = [col.replace('"', '') for col in datas.columns]


# features = ['  exx', '  eyy']
# x = np.array(datas.loc[:, features])
# x = np.transpose(x)
# X = torch.from_numpy(x)
# print(X)
class mydataset(Dataset):
    def __init__(self, x, filepath,):
        datas = pd.read_csv(filepath, dtype=float)
        datas.columns = [col.replace('"', '') for col in datas.columns]

        x = np.array(datas.loc[:, features])
        x = np.transpose(x)
        X = torch.from_numpy(x)
