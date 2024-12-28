import argparse
import yaml
import models
from dataset import VAEDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


def train():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Trainer for VAE')
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE',
                        default='configs/vanilla_vae.yaml',
                        help='path to the config file')
    parser.add_argument('--checkpoint', '-p', dest="checkpoint", metavar='FILE',
                        default=None, help='path to model checkpoint')
    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        config = yaml.safe_load(file)

    tb_logger = TensorBoardLogger(save_dir='logs/', name='VanillaVAE')
    csv_logger = CSVLogger(save_dir='logs/', name='VanillaVAE')

    encoder = models.Encoder(
        in_channels=config['model_params']['in_channels'],
        z_dim=config['model_params']['z_dim']
    )
    decoder = models.Decoder(z_dim=config['model_params']['z_dim'])
    vanilla_vae = models.VanillaVAE(
        lr=config['training_params']['lr'],
        beta=config['training_params']['kl_weight'],
        encoder=encoder,
        decoder=decoder
    )

    if args.checkpoint:
        print('Loading model from checkpoint.')
        vanilla_vae = models.VanillaVAE.load_from_checkpoint(
            args.checkpoint,
            lr=config['training_params']['lr'],
            beta=config['training_params']['kl_weight'],
            encoder=encoder,
            decoder=decoder
        )()

    data = VAEDataset(
        data_path=config['data_params']['data_path'],
        batch_size=config['data_params']['batch_size'],
        num_workers=config['data_params']['num_workers'],
        pin_memory=config['data_params']['pin_memory']
    )
    data.setup()

    trainer = Trainer(
        logger=tb_logger,
        accelerator=config['hardware_params']['accelerator'],
        devices=config['hardware_params']['devices'],
        max_epochs=config['training_params']['max_epochs']
    )
    trainer.fit(model=vanilla_vae, datamodule=data)

if __name__ == '__main__':
    train()
