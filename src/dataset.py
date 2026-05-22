"""Dataset для классификации AI vs Human Generated Images."""
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Читает CSV (file_name, label) и подтягивает картинки из root_dir.

    Колонка file_name в CSV всегда содержит префикс 'train_data/...', но реальная
    папка в Test_1/Test_2 называется 'test_data'. Поэтому берём basename и сами
    подбираем правильный подкаталог.
    """

    def __init__(self, csv_file, root_dir, transform=None, subsample: int = 0, seed: int = 42):
        df = pd.read_csv(csv_file)
        if subsample and subsample > 0 and subsample < len(df):
            df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)
        self.data = df
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Подбираем подкаталог с изображениями (train_data или test_data)
        for sub in ("train_data", "test_data"):
            if (self.root_dir / sub).is_dir():
                self.image_subdir = self.root_dir / sub
                break
        else:
            raise FileNotFoundError(f"Не нашёл train_data/test_data в {self.root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel = self.data.iloc[idx]["file_name"]
        # Берём только basename — игнорируем некорректный префикс в CSV
        img_path = self.image_subdir / Path(rel).name
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_test_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
