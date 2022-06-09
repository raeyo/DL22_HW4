import timm
import torch
from kimchi_dataset import KimChiData
from torch.utils.data import DataLoader
import os

if __name__=="__main__":
    #TODO: change data root and checkpoint
    data_root = '/data/datasets/한국 음식 이미지/김치'
    ckp = "ckp/best.pth"
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    model = timm.create_model('resnet50', pretrained=True, num_classes=11, in_chans=3)
    model.to(device)
    model.load_state_dict(torch.load(ckp))
    
    test_dataset = KimChiData(data_root=data_root, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    total = 0
    correct = 0
    
    model.eval()
    for b, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        total += inputs.shape[0]
        _, predicted = outputs.max(1)
        correct += torch.eq(targets, predicted).sum().item()
    
    acc = correct / total
    print("Test Acc: {}".format(acc))
    