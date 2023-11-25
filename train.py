import models
from dataset import VAEDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

tb_logger = TensorBoardLogger(save_dir='logs/', name='Vanilla VAE')    

encoder = models.Encoder(in_channels=3, z_dim=512)
decoder = models.Decoder(z_dim=512)
vanilla_vae = models.VanillaVAE(lr=1e-4,
                                beta=1e-4,
                                encoder=encoder, 
                                decoder=decoder)

"""
A work-around to address issues with pytorch's celebA dataset class.
   
Download and Extract
URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
"""
data = VAEDataset(batch_size=128,
                  num_workers=16,
                  pin_memory=True)
data.prepare_data()
data.setup()

trainer = Trainer(logger=tb_logger,
                  gpus=[1],
                  max_epochs=30)
trainer.fit(model=vanilla_vae, datamodule=data)
