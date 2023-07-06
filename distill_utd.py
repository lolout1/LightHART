import torch
import numpy as np
from Make_Dataset import Poses3d_Dataset, Utd_Dataset
import torch.nn as nn
import PreProcessing_ncrc
from Models.model_crossview_fusion import ActTransformerMM
from Models.model_acc_only import ActTransformerAcc
# from Tools.visualize import get_plot
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from arguments import parse_args
import pickle
from asam import ASAM, SAM
# from timm.loss import LabelSmoothingCrossEntropy
from loss import SemanticLoss, FocalLoss
import os


#Define loss and optimizer
#Learning rate decay 




#Learning Rate Scheduler
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer,(num_epochs)
#print("Using cosine")

#TRAINING AND VALIDATING


#Label smoothing
#smoothing=0.1
#criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
#print("Loss: LSC ",smoothing)



def train(epoch, num_epochs, student_model, teacher_model, criterion, best_accuracy,):
    teacher_model.eval()
    with tqdm(total  = len(training_generator), desc = f'Epoch {epoch}/{num_epochs}',ncols = 128) as pbar:
        # Train
        student_model.train()
        train_loss = 0.
        accuracy = 0.
        teacher_accuracy = 0.
        cnt = 0.

        for inputs, targets in training_generator:
            # Transfering the input, targets to the GPU]
            inputs = inputs.to(device) #[batch_size X ]
            targets = targets.to(device)
            optimizer.zero_grad()

            #Prediction step
            out, student_logits,student_pred = student_model(inputs.float())
            teacher_out, teacher_logits, teacher_pred = teacher_model(inputs.float())
            # print(f'\nStudent logit shape: {student_logits.shape}')
            # print(f'\nTeacher logit shape: {teacher_logits.shape}')
        
            loss = criterion(student_logits, targets, teacher_logits)
            loss.mean().backward()
            optimizer.step()
            # minimizer.ascent_step()

            # # Descent Step
            # # Not really sure why I need to make prediction again 
            # _, des_student_logits,des_stud_pred = student_model(inputs.float())
            # _, des_teacher_logits, des_teacher_pred = teacher_model(inputs.float())
            # criterion(des_student_logits, targets, des_teacher_logits, T=2.0, alpha = 0.7).mean().backward()
            # minimizer.descent_step()

            with torch.no_grad():
                train_loss += loss.sum().item()
                accuracy += (torch.argmax(student_pred, 1) == targets).sum().item()
                teacher_accuracy += (torch.argmax(teacher_pred, 1) == targets).sum().item()
            cnt += len(targets)

            temp_loss = train_loss / cnt
            temp_acc = (accuracy * 100) / cnt
            temp_teach_acc = (teacher_accuracy * 100) / cnt
            pbar.update(1)
            pbar.set_postfix({'train_loss' : temp_loss, 'train_acc' : temp_acc, 'teacher_acc': temp_teach_acc})
            
        train_loss /= cnt
        accuracy *= 100. / cnt
        epoch_loss_train.append(train_loss)
        epoch_acc_train.append(accuracy)

        #Val
        student_model.eval()
        val_loss = 0
        val_accuracy = 0
        val_t_accuracy = 0
        cnt = 0.
        student_model=student_model.to(device)
        with torch.no_grad():
            for inputs,targets in validation_generator:
                inputs = inputs.to(device); #print("Validation input: ",inputs)
                targets = targets.to(device)
                
                out, student_logits,predictions = student_model(inputs.float())
                teacher_out, teacher_logits, teacher_pred = teacher_model(inputs.float())
                loss_score = FocalLoss()(predictions, targets)
                
                with torch.no_grad():
                    val_loss += loss_score.sum().item()
                    val_accuracy += (torch.argmax(predictions, 1) == targets).sum().item()
                    val_t_accuracy += (torch.argmax(teacher_pred, 1) == targets).sum().item()
                cnt += len(targets)
            val_loss /= cnt
            val_accuracy *= 100. / cnt
            val_t_accuracy *= 100./ cnt
        print(f"\n Epoch: {epoch},Val accuracy:  {val_accuracy:6.2f} %, Val loss:  {val_loss:8.5f}% , Val_teach: {val_t_accuracy:8.5f}%")
        if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                #need to add arguements here also for different experiments
                torch.save(student_model.state_dict(),PATH+exp+'kd_wfocal.pt'); 
                print("Check point "+PATH+exp+'kd_wfocal.pt'+ ' Saved!')


        epoch_loss_val.append(val_loss)
        epoch_acc_val.append(val_accuracy)
        scheduler.step()
        return best_accuracy



if __name__ == "__main__":

    args = parse_args()
    max_epoch = args.epochs
    best_accuracy = 0 
    epoch_loss_train=[]
    epoch_loss_val=[]
    epoch_acc_train=[]
    epoch_acc_val=[]

    exp = 'utdKD' #Assign an experiment id
    dataset = 'utd'
    mocap_frames = 100
    acc_frames = 150
    num_joints =20
    num_classes = 27
    lr=0.01
    wt_decay=5e-3

    if not os.path.exists('exps/'+exp+'/'):
        os.makedirs('exps/'+exp+'/')
    PATH='exps/'+exp+'/'

    #CUDA for PyTorch
    print("Using CUDA....")

    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True



    # Parameters
    print("Creating params....")
    params = {'batch_size':16,
            'shuffle': True,
            'num_workers': 0}

    num_epochs = 100

    # Generators
    #pose2id,labels,partition = PreProcessing_ncrc_losocv.preprocess_losocv(8)

    if dataset == 'ncrc':
        tr_pose2id,tr_labels,valid_pose2id,valid_labels,pose2id,labels,partition = PreProcessing_ncrc.preprocess()
        training_set = Poses3d_Dataset( data='ncrc',list_IDs=partition['train'], labels=tr_labels, pose2id=tr_pose2id, mocap_frames=mocap_frames, acc_frames=acc_frames,has_features=True, normalize=False)
        training_generator = torch.utils.data.DataLoader(training_set, **params) #Each produced sample is  200 x 59 x 3

        validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['valid'], labels=valid_labels, pose2id=valid_pose2id, mocap_frames=mocap_frames,has_features=True, acc_frames=acc_frames ,normalize=False)
        validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3

    else:
        training_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/randtrain_data.npz')
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        validation_set = Utd_Dataset('/home/bgu9/Fall_Detection_KD_Multimodal/data/UTD_MAAD/randvalid_data.npz')
        validation_generator = torch.utils.data.DataLoader(validation_set, **params)


    #Define model
    print("Initiating Model...")
    teacher_model = ActTransformerMM(device = device, mocap_frames=mocap_frames, acc_frames=150, num_joints=num_joints, in_chans=3, acc_coords=3,
                                    acc_features=1, spatial_embed=32,has_features = False,num_classes=num_classes,embed_type='conv')

    #Define model
    print("Initiating Model...")

    student_model = ActTransformerAcc(adepth = 3,num_classes=num_classes,device= device, acc_frames= acc_frames, num_joints = 20,has_features=False, num_heads=2, acc_embed = 32)

    teacher_model.to(device=device)
    student_model.to(device=device)

    teacher_model.load_state_dict(torch.load('/home/bgu9/Fall_Detection_KD_Multimodal/exps/teacher-utd/utd_best_ckptconv.pt', map_location=torch.device('cpu')))
    #Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr,weight_decay=wt_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min= 2.5e-05,verbose=True)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wt_decay)

    #ASAM
    # rho=0.5
    # eta=0.01
    # minimizer = ASAM(optimizer, student_model, rho=rho, eta=eta)
    
    best_accuracy = 0
    criterion = SemanticLoss(alpha = 0.7,T = 2.0)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 7)
    #criterion selection using arguements
    total_params = 0
    print("-----------TRAINING PARAMS----------")
    for name , params in student_model.named_parameters():
        total_params += params.numel()
        print(f'Layer {name} | Size: {params.size()} | Params: {params.numel()}')
    print(f'Total parameter: {total_params}')
    
    for epoch in range(1,max_epoch+1): 
        best_acc = train(epoch, max_epoch, student_model, teacher_model,criterion,  best_accuracy = best_accuracy)
        best_accuracy = best_acc