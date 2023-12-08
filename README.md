|Reconstruction | Samples |
|---------------|---------|
|    ![][2]     | ![][1]  |

### Sample Interpolation

![][3]

### Installation
```
$ git clone https://github.com/Bex0n/CelebA-VAE
$ cd CelebA-VAE
$ pip install -r requirements.txt
```

### Usage
```
cd CelebA-VAE
python train.py -c <path_to_config_file>
```

**Note:** model capabilities such as generating new celebrities or modifying features are available in the file experiments/vanilla_vae.py

### Citation
```
@misc{Bex0n,
  author = {Dawidowicz, J.},
  title = {CelebA-VAE},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Bex0n/CelebA-VAE}}
}
```

[1]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_training_data.png
[2]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_reconstruction.png
[3]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_interpolation.png