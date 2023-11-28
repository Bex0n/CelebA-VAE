import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class ImageFolderNoLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_folder = ImageFolder(root=self.root, transform=self.transform)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, index):
        image, _ = self.image_folder[index]
        return image


def display_random(dataset, num):
    plt.figure(figsize=(16, 8))

    columns = 5
    rows = num // columns + 2
    gs = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.2)

    for idx in range(num):
        img = dataset[idx].permute(1, 2, 0)

        ax = plt.subplot(gs[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow((img * 255).int())

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()


def display_img(images):
    plt.figure(figsize=(16, 8))

    num = images.size()[0]
    columns = 5
    rows = num // columns + 2
    gs = gridspec.GridSpec(rows, columns, wspace=0.0, hspace=0.1)

    for idx in range(num):
        img = images[idx].permute(1, 2, 0).to('cpu')

        ax = plt.subplot(gs[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow((img * 255).int())

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
