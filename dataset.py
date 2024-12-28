from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import pytorch_lightning as L
import os

class CelebADataset(Dataset):
    def __init__(self, images_dir, attr_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.attr_data = self._load_attr_data(attr_file)
        
    def _load_attr_data(self, attr_file):
        # Load the attribute file and parse the labels
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        # Parse the header and skip the first two lines
        header = lines[1].strip().split()
        data = [line.strip().split() for line in lines[2:]]
        df = pd.DataFrame(data, columns=['image'] + header)
        # Convert labels to integers
        for col in header:
            df[col] = df[col].astype(int)
        return df

    def __len__(self):
        return len(self.attr_data)

    def __getitem__(self, idx):
        row = self.attr_data.iloc[idx]
        img_path = os.path.join(self.images_dir, row['image'])
        image = Image.open(img_path).convert('RGB')
        label = row['Male']
        if self.transform:
            image = self.transform(image)
        return image, label

class VAEDataset(L.LightningDataModule):
    def __init__(self,
                 data_path: str = None,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()

        if data_path is None:
            data_path = os.getcwd()
        self.data_dir = data_path
        self.batch_szie = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _load_attr_data(self, attr_file):
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        # Parse the header and skip the first two lines
        header = lines[1].strip().split()
        data = [line.strip().split() for line in lines[2:]]
        df = pd.DataFrame(data, columns=['image'] + header)
        # Convert labels to integers
        for col in header:
            df[col] = df[col].astype(int)
        return df

    def setup(self, stage=None) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128, antialias=True),
            transforms.CenterCrop(128),
        ])

        try:
            self.celeba_train = CelebADataset(
                images_dir=Path(self.data_dir) / "img_align_celeba",
                attr_file=Path(self.data_dir) / "list_attr_celeba.txt",
                transform=transform
            )
        except Exception:
            print('\033[91m',
                  'The CelebA dataset is missing.\n'
                  'Download and extract file below following the instructions in README: \n'
                  'https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing\n', 
                  '\033[0m')
            raise FileNotFoundError(
                f'CelebA dataset not found in {self.data_dir}. '
                'Make sure it contains another directory with images inside.'
            )

    def train_dataloader(self) -> DataLoader:
        celeba_train = DataLoader(self.celeba_train,
                                  batch_size=self.batch_szie,
                                  pin_memory=self.pin_memory,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        return celeba_train
