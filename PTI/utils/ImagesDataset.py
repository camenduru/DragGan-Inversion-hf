import os

from torch.utils.data import Dataset
from PIL import Image

from PTI.utils.data_utils import make_dataset
from torchvision import transforms


class Image2Dataset(Dataset):
    def __init__(self, image) -> None:
        super().__init__()
        self.image = image
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return "customIMG", self.transform(self.image)


class ImagesDataset(Dataset):
    def __init__(self, source_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert("RGB").resize([1024, 1024])

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im
