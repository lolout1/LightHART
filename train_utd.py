import torch
import numpy as np
import pandas as pd
from Make_Dataset import Poses3d_Dataset, Utd_Dataset, UTD_mm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import PreProcessing_ncrc
from Models.earlyconcat import ActTransformerAcc
from Models.model_crossview_fusion import ActTransformerMM
from Models.model_simple_fusion import ActRecogTransformer
from Models.earlyfusion import MMTransformer
#from Models.model_acc_only import ActTransformerAcc
from loss import FocalLoss
# from Tools.visualize import get_plot
import pickle
from asam import ASAM, SAM
# from timm.loss import LabelSmoothingCrossEntropy
import os

exp = 'utd' #Assign an experiment id

if not os.path.exists('exps/'+exp+'/'):
    os.makedirs('exps/'+exp+'/')
PATH='exps/'+exp+'/'

#CUDA for PyTorch
print("Using CUDA....")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':4,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 200

# Generators
#pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)
# pose2id, labels, partition = PreProcessing_ncrc.preprocess()

print("Creating Data Generators...")
#arguments
dataset = 'utd'
mocap_frames = 100
acc_frames = 100
num_joints = 20 
num_classes = 27

#hyperparameters
lr = 0.001
wt_decay=1e-3

if dataset == 'ncrc':
    tr_pose2id,tr_labels,valid_pose2id,valid_labels,pose2id,labels,partition = PreProcessing_ncrc.preprocess()
    training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'],has_features= False,labels=tr_labels, pose2id=tr_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames, normalize=True)
    training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

    validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['valid'],has_features = False,labels=valid_labels, pose2id=valid_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=True)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

    test_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels,has_features = False,pose2id=pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames ,normalize=False)
    test_generator = torch.utils.data.DataLoader(test_set, **params) #Each produced sample is 6000 x 229 x 3

else:
    training_set = UTD_mm('data/UTD_MAAD/utd_train100.npz', batch_size=params['batch_size'])
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = UTD_mm('data/UTD_MAAD/utd_valid100.npz',  batch_size=params['batch_size'])
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)


#Define model
print("Initiating Model...")
# model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=acc_frames, num_joints=num_joints, in_chans=3, acc_coords=3,
#                                   acc_features=1, spatial_embed=16,has_features = False,num_classes=num_classes, num_heads=8)
model = ActTransformerAcc(acc_embed=16,adepth = 2,device= device, acc_frames= acc_frames, num_joints = num_joints,has_features=False, num_heads = 2, num_classes=num_classes)
#model = MMTransformer(device=device, mocap_frames=mocap_frames, acc_frames=acc_frames,num_joints=num_joints,num_classes=num_classes)
#model= ActRecogTransformer(device = device, mocap_frames=mocap_frames, acc_frames=acc_frames, num_classes = 27, num_joints=num_joints)
model = model.to(device)


print("-----------TRAINING PARAMS----------")
#Define loss and optimizer

# class_weights = torch.reciprocal(torch.tensor([74.23, 83.87, 56.75, 49.78, 49.05, 93.92]))
criterion = torch.nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=0.25, gamma=2)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)
# scheduler = CosineAnnealingLR(optimizer=optimizer,T_max=100, eta_min=1e-4,last_epoch=-1,verbose=True)

#ASAM
rho=0.5
eta=0.01
# minimizer = ASAM(optimizer, model, rho=rho, eta=eta)

#Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, max_epochs)
#print("Using cosine")

#TRAINING AND VALIDATING
epoch_loss_train=[]
epoch_loss_val=[]
epoch_acc_train=[]
epoch_acc_val=[]

#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
#print("Loss: LSC ",smoothing)

best_accuracy = 0
# model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/myexp-utd/myexp-utd_best_ckptutdmm.pt'))
# scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 10)

print("Begin Training....")
for epoch in range(max_epochs):
    # Train
    model.train()
    loss = 0.
    accuracy = 0.
    cnt = 0.
    pred_list = []
    target_list = []
    for inputs, targets in training_generator:
        inputs = inputs.to(device); #print("Input batch: ",inputs)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Ascent Step
        #print("labels: ",targets)
        out, logits,predictions = model(inputs.float())
        #print("predictions: ",torch.argmax(predictions, 1) )
        batch_loss = criterion(logits, targets)
        batch_loss.mean().backward()
        optimizer.step()
        # minimizer.ascent_step()

        # # Descent Step
        # _,_,des_predictions = model(inputs.float())
        # criterion(des_predictions, targets).mean().backward()
        # minimizer.descent_step()
        pred_list.extend(torch.argmax(predictions, 1))
        target_list.extend(targets)
        with torch.no_grad():
            loss += batch_loss.sum().item()
            accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
        cnt += len(targets)
    loss /= cnt
    accuracy *= 100. / cnt
    # print('---Train---')
    # for item1 , item2 in zip(target_list, pred_list):
    #     print(f'{item1} | {item2}')
    # scheduler.step()
    print(f"Epoch: {epoch}, Train accuracy: {accuracy:6.2f} %, Train loss: {loss:8.5f}")
    epoch_loss_train.append(loss)
    epoch_acc_train.append(accuracy)
    #scheduler.step()

    # accuracy,loss = validation(model,validation_generator)
    # Test
    model.eval()
    val_loss = 0.
    val_accuracy = 0.
    cnt = 0.
    val_pred_list = []
    val_trgt_list = []
    # model=model.to(device)
    with torch.no_grad():
        for inputs, targets in validation_generator:
            b = inputs.shape[0]
            inputs = inputs.to(device); #print("Validation input: ",inputs)
            targets = targets.to(device)
            
            _,logits,predictions = model(inputs.float())
            val_pred_list.extend(torch.argmax(predictions, 1))
            val_trgt_list.extend(targets)

            loss = criterion(logits,targets)
            val_loss += loss.sum().item()
            val_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
            cnt += len(targets)
        val_loss /= cnt
        val_accuracy *= 100. / cnt
        # print('---Val---')
        # for item1 , item2 in zip(val_trgt_list, val_pred_list):
        #         print(f'{item1} | {item2}')
        if best_accuracy < val_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(),PATH+'utdmmd4h4_woKD_ef.pt')
            print("Check point "+PATH+'utdmmd4h4_woKD_ef.pt'+ ' Saved!')

    print(f"Epoch: {epoch},Valid accuracy:  {val_accuracy:6.2f} %, Valid loss:  {val_loss:8.5f}")
    epoch_loss_val.append(val_loss)
    epoch_acc_val.append(accuracy)


data_dict = {'train_accuracy': epoch_acc_train, 'train_loss':epoch_loss_train, 'val_acc': epoch_acc_val , 'val_loss' :  epoch_loss_val }
df = pd.DataFrame(data_dict)
df.to_csv('loss_utdmmd4h4_woKD_ef.csv')

print(f"Best test accuracy: {best_accuracy}")
print("TRAINING COMPLETED :)")

#Save visualization
# get_plot(PATH,epoch_acc_train,epoch_acc_val,'Accuracy-'+exp,'Train Accuracy','Val Accuracy','Epochs','Acc')
# get_plot(PATH,epoch_loss_train,epoch_loss_val,'Loss-'+exp,'Train Loss','Val Loss','Epochs','Loss')
