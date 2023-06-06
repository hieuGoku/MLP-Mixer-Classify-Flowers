from torch.utils.data import DataLoader, Dataset
from PIL import Image

class dataset(Dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

        self.class_map = {"daisy" : 0, "dandelion": 1,
            "roses" : 2, "sunflowers": 3, "tulips" : 4}
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)

        img_transform = self.transforms(img)
        class_id = self.class_map[img_path.split("/")[-2]]
        return img_transform, class_id