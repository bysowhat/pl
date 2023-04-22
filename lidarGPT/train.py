# from simclr.datamodule.stl10 import STL10
import torch
from torchvision import transforms

from datasets.stl10 import STL10Dataset
from utils.show import show_2dimg

class Mytrans(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        Mytrans(),
    ]
)
ds = STL10Dataset('/mnt/c/Users/a/Datasets/stl10/stl10_binary', 
                  split='unsv', 
                  transform=train_transforms)

for data in ds:
    img, _ = data
    show_2dimg(img)
    print(1)