import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch
import numpy as np
from PIL import Image
class KimChiData(Dataset):
    class2label = {'갓김치': 0, '깍두기': 1, '나박김치': 2, '무생채': 3, '배추김치': 4, '백김치': 5, '부추김치': 6, '열무김치': 7, '오이소박이': 8, '총각김치': 9, '파김치': 10}
    label2class = {0: '갓김치', 1: '깍두기', 2: '나박김치', 3: '무생채', 4: '배추김치', 5: '백김치', 6: '부추김치', 7: '열무김치', 8: '오이소박이', 9: '총각김치', 10: '파김치'}
    def __init__(self, data_root, split='train'):
        self.img_list = []
        self.label_list = []
        self.split = split
        data_info = {k: 0 for k in self.class2label.keys()}
        print("Data split: {}".format(split))
        
        if self.split == 'train':
            self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        temp = {}
        for k, v in self.class2label.items():
            cls_dir = os.path.join(data_root, k)
            img_list = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)]
            for img_file in img_list:
                if self.check_split(img_file) != self.split:
                    continue
                try:
                    img = Image.open(img_file)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                except:
                    print("Can not load {}".format(img_file))
                    continue
                data_info[k] += 1
                self.img_list.append(img_file)
                self.label_list.append(v)
        print(data_info)
    
    @staticmethod
    def check_split(img_path):
        try:
            img_idx = int(os.path.splitext(img_path)[0].split("_")[-1])
        except:
            return "NO"
        if 0 <= img_idx <= 800: # train
            return "train"
        elif 900 <= img_idx : # test
            return "test"
        else:
            return "val"
        
    
    def __len__(self):
        return len(self.label_list)
        
    def __getitem__(self, index):
        img_file = self.img_list[index]
        img = Image.open(img_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.label_list[index]
        
        return self.transform(img), torch.tensor(label)
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    data_root = '/data/datasets/한국 음식 이미지/김치'
    tr = KimChiData(data_root, split='val')
    for img, label in tr:
        plt.imshow(img.permute(1, 2, 0))