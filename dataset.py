from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import pytorch_lightning as L
import os


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

    def prepare_data(self) -> None:
        try:
            CelebA(self.data_dir, download=True)
        except RuntimeError:
            print('\033[91m',
                  'Pytorch CelebA dataset download is blocked due to daily google drive download limit.\n'
                  'Download and extract file below. Also comment out prepare_data() step: \n'
                  'https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing\n', 
                  '\033[0m')

    def setup(self, stage=None) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(128, antialias=True),
            transforms.CenterCrop(128),
        ])

        self.celeba_train = CelebA(self.data_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        celeba_train = DataLoader(self.celeba_train,
                                  batch_size=self.batch_szie,
                                  pin_memory=self.pin_memory,
                                  num_workers=self.num_workers,
                                  shuffle=True)
        return celeba_train
