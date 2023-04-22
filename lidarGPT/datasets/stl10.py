import os
import os.path as osp
from typing import Any, Callable, cast, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class STL10Dataset(Dataset):
    def __init__(self, root, split, transform):
        self.root = root
        self.split = split
        self.transform = transform
        assert split in ['unsv', 'train_cls', 'val_cls', 'test_cls']

        fns = os.listdir(self.root)
        assert 'unlabeled_X.bin' in fns
        assert 'train_y.bin' in fns
        assert 'train_X.bin' in fns
        assert 'test_y.bin' in fns
        assert 'test_X.bin' in fns
        assert 'class_names.txt' in fns

        # Assign train/val datasets for use in dataloaders
        if split == 'unsv':
            self.data = self._loadfile('unlabeled_X.bin')
        elif split in ['train_cls', 'val_cls']:
            x, y = self._loadfile('train_X.bin', 'train_y.bin')#(nchw,n)
            train_num = int(x.shape[0] *0.9)
            if split == 'train_cls':
                x = x[:train_num]
                y = y[:train_num]
                self.data = (x,y)
            elif split == 'train_cls':
                x = x[train_num:]
                y = y[train_num:]
                self.data = (x,y)
        elif split == 'test_cls':
            self.data = self._loadfile('test_X.bin', 'test_y.bin')#(nchw,n)
        
        class_file = osp.join(self.root, 'class_names.txt')
        with open(class_file) as f:
            self.classes = f.read().splitlines()

    def _loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = osp.join(self.root, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = osp.join(self.root, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels
 
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.data[0][index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img = self.transform(img)
        
        if self.data[1] is not None:
            return img, self.data[1][index]
        else:
            return img, None