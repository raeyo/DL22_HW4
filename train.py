import timm
import torch
from kimchi_dataset import KimChiData
from torch.utils.data import DataLoader
import os

if __name__=="__main__":
    data_root = '/data/datasets/한국 음식 이미지/김치'
    ckp_dir = "ckp"
    
    tr_dataset = KimChiData(data_root=data_root, split='train')
    val_dataset = KimChiData(data_root=data_root, split='val')
    
    tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    model = timm.create_model('resnet50', pretrained=True, num_classes=11, in_chans=3)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)
    
    best_acc = 0
    best_epoch = 0
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for b, (inputs, targets) in enumerate(tr_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()            
            optimizer.step()
            
            total_loss += loss
            total += inputs.shape[0]
            _, predicted = outputs.max(1)
            correct += torch.eq(targets, predicted).sum().item()
            
            optimizer.zero_grad()
            
        acc = correct / total
        loss = total_loss / (b+1)
        print("Train Epoch: {:<3} | Acc: {:<5.2f} | Loss: {:<5.2f}".format(epoch, acc, loss))
        
        
        # validation
        total_loss = 0
        total = 0
        correct = 0
        
        model.eval()
        for b, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            total_loss += loss
            total += inputs.shape[0]
            _, predicted = outputs.max(1)
            correct += torch.eq(targets, predicted).sum().item()
        
        acc = correct / total
        loss = total_loss / (b+1)
        print("Val Epoch: {:<3} | Acc: {:<3} | Loss: {:<3}".format(epoch, acc, loss))
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckp_dir, "ep_{}.pth".format(epoch)))
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckp_dir, "best.pth"))
            print("Best Epoch: {:<3} | Best Acc: {:<3}".format(best_epoch, best_acc))
        