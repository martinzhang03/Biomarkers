import torch
import pandas as pd
from PIL import Image as img
import torchvision.transforms as t
import constants

class StartingDataset(torch.utils.data.Dataset):
    """
    Load 800 x 600 images
    """

    def __init__(self, csv_path, im_path, training_set=True):
        df = pd.read_csv(csv_path)
        self.image_id = df['img_index']
        self.labels = df['age']
        self.im_path = im_path
        self.training_set = training_set

        if training_set:
            self.image_id = self.image_id[:constants.TRAIN_NUM]
            self.labels = self.labels[:constants.TRAIN_NUM]
        else:
            self.image_id = self.image_id[:constants.TEST_NUM]
            self.labels = self.labels[:constants.TEST_NUM]

    def __getitem__(self, index):
        id = self.image_id.iloc[index]
        label = torch.tensor(int(self.labels.iloc[index]), dtype=torch.float)  # No adjustment

        if self.training_set:
            img_path = constants.TRAIN_IMG_PATH + id
        else:
            img_path = constants.TEST_IMG_PATH + id

        # image = img.open(img_path).convert('RGB')
        image = img.open(img_path)
        image = image.convert('L')
        image = image.resize((28, 28), img.BILINEAR)
        # return((image), label)
        return (t.ToTensor()(image), label)

    def __len__(self):
        return len(self.labels)
