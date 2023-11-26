import argparse
import models
import yaml
from dataset import VAEDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchinfo import summary


parser = argparse.ArgumentParser(description='Trainer for VAE')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vanilla_vae.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(e)

tb_logger = TensorBoardLogger(save_dir='logs/', name='VanillaVAE')

encoder = models.Encoder(in_channels=config['model_params']['in_channels'],
                         z_dim=config['model_params']['z_dim'])
decoder = models.Decoder(z_dim=config['model_params']['z_dim'])
vanilla_vae = models.VanillaVAE(lr=config['training_params']['lr'],
                                beta=config['training_params']['kl_weight'],
                                encoder=encoder,
                                decoder=decoder)

"""
A work-around to address issues with pytorch's celebA dataset class.

Download and Extract
URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
"""
data = VAEDataset(data_path=config['data_params']['data_path'],
                  batch_size=config['data_params']['batch_size'],
                  num_workers=config['data_params']['num_workers'],
                  pin_memory=config['data_params']['pin_memory'])
# data.prepare_data()
data.setup()

trainer = Trainer(logger=tb_logger,
                  accelerator=config['hardware_params']['accelerator'],
                  devices=config['hardware_params']['devices'],
                  max_epochs=config['training_params']['max_epochs'])
trainer.fit(model=vanilla_vae, datamodule=data)
