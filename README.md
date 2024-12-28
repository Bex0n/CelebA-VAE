# CelebA-VAE

<table>
  <tr>
    <td style="width:50%">
      <img src="https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_reconstruction.png" width="100%"/>
    </td>
    <td style="width:50%">
      <img src="https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_training_data.png" width="100%"/>
    </td>
  </tr>
</table>

![][3]

### Installation
```
$ git clone https://github.com/Bex0n/CelebA-VAE
$ cd CelebA-VAE
$ pip install -r requirements.txt
```

Download the dataset [link](https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing) and extract:

```
unzip celeba.zip
cd celeba
unzip img_align_celeba.zip
cd ..
```

### Usage
```
python train.py -c <path_to_config_file>
```
If you want to train the model with cpu only, change `accelerator: 'auto'` to `accelerator: 'cpu'` in the provided config file.

**Note:** model capabilities such as generating new celebrities or modifying features are available in the file experiments/vanilla_vae.py

[1]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_training_data.png
[2]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_reconstruction.png
[3]: https://raw.githubusercontent.com/Bex0n/CelebA-VAE/main/assets/vae_interpolation.png
