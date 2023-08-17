from torch.utils.data import Dataset
from PIL import Image
import os

import torchvision.transforms as transforms
class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = self.root_dir
        self.img_path = os.listdir(self.path)
 
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path).convert('L')
        img_tensor = transforms.ToTensor()(img.resize((64,48)))
        return img_tensor
    def __len__(self):
        return len(self.img_path)
 


# root_dir = "dataset/train"
# mydataset = MyData(root_dir)
#  # "len(train_dataset)"指令可以在Python console中查看train_dataset数据集中有多少个元素。
# for i in range(230):
#     img = mydataset[i]
# # img.show()
# print("ok")