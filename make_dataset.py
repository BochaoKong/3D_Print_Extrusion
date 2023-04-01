# construct a dataloader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, meta_data, img_size):
        """Set the path for images and classification results.
        Args:
            root: image directory.
            meta_data: data frame with image path and label.
            img_size: image size (after resize), a tuple (width, height).
        """
        self.root = root
        self.meta = meta_data
        self.img_size = img_size


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        path = self.meta['img_path'][index]
        target = self.meta['has_under_extrusion'][index]

        image = Image.open(os.path.join(self.root, path))
        #image = image.convert('L')

        n, m = self.img_size

        transform = transforms.Compose([
            transforms.Resize( (n, m) ),
            transforms.ToTensor(),
        ])
        image = transform(image)

        return image, target
    

    def __len__(self):
        return self.meta.shape[0]