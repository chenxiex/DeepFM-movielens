import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from torchfm.dataset.movielens import MovieLens1MDataset
from torchfm.model.dfm import DeepFactorizationMachineModel

class EarlyStopper(object):
    
    def __init__(self,num_trials,save_path):
        self.num_trials=num_trials
        self.trial_counter=0
        self.best_accuracy=0
        self.save_path=save_path
    
    def is_continuable(self,model,accuracy):
        if accuracy>self.best_accuracy:
            self.best_accuracy=accuracy
            self.trial_counter=0
            torch.save(model,self.save_path)
            return True
        elif self.trial_counter<self.num_trials:
            self.trial_counter+=1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(data_loader):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()

def test(model,data_loader,device):
    model.eval()
    targets,predicts=list(),list()
    with torch.no_grad():
        for fields,target in data_loader:
            fields,target=fields.to(device),target.to(device)
            y=model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets,predicts)

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=MovieLens1MDataset('./data/ml-1m/ratings.dat')
    train_length=int(len(dataset)*0.8)
    valid_length=int(len(dataset)*0.1)
    test_length=len(dataset)-train_length-valid_length
    train_dataset,valid_dataset,test_dataset=torch.utils.data.random_split(dataset,(train_length,valid_length,test_length))
    train_data_loader=DataLoader(train_dataset,batch_size=2048,num_workers=8)
    valid_data_loader=DataLoader(valid_dataset,batch_size=2048,num_workers=8)
    test_data_loader=DataLoader(test_dataset,batch_size=2048,num_workers=8)
    model=DeepFactorizationMachineModel(dataset.field_dims,embed_dim=16,mlp_dims=(16,16),dropout=0.2).to(device)
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(params=model.parameters(),lr=0.001,weight_decay=1e-6)
    early_stopper=EarlyStopper(num_trials=2,save_path=f'./dfm.pt')
    for epoch_i in range(100):
        train(model,optimizer,train_data_loader,criterion,device)
        auc=test(model,valid_data_loader,device)
        if (not early_stopper.is_continuable(model,auc)):
            break
    auc=test(model,test_data_loader,device)
    print(f'auc={auc}')

if __name__=='__main__':
    main()