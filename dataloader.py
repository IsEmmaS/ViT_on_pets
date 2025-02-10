import torch
import os
import random

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class myDataset(Dataset):

    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.class_to_idx = self._create_class_to_idx()
        # print(self.class_to_idx)

    def __len__(self):
        return len(self.file_paths)

    def _create_class_to_idx(self):
        """创建标签到索引的映射"""
        classes = set()
        for filename in self.file_paths:
            filename_without_ext = os.path.splitext(filename)[0]
            label = filename_without_ext.rsplit("_", 1)[0]
            classes.add(label)
        # print(classes)
        sorted_classes = sorted(classes)
        return {cls: idx for idx, cls in enumerate(sorted_classes)}

    def __getitem__(self, index):
        filename = self.file_paths[index]
        filename_without_ext = os.path.splitext(filename)[0]
        label = filename_without_ext.rsplit("_", 1)[0]
        label_index = self.class_to_idx[label]

        image = datasets.folder.default_loader(filename)
        if self.transform is not None:
            image = self.transform(image)

        return image, label_index


def create_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4):
    
    all_files = [
        f for f in os.listdir(data_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    all_file_paths = [os.path.join(data_dir, f) for f in all_files]

    
    random.seed(37)
    random.shuffle(all_file_paths)
    train_size = int(0.8 * len(all_file_paths))
    train_files = all_file_paths[:train_size]
    val_files = all_file_paths[train_size:]

    # 定义数据变换
    # 更新数据增强策略
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=3, magnitude=15),  # 强化增强
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5) 
    ])
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = myDataset(train_files, transform=train_transform)
    val_dataset = myDataset(val_files, transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":

    data_dir = "images"
    train_loader, val_loader = create_dataloaders(data_dir)
    print(len(train_loader), len(val_loader))
