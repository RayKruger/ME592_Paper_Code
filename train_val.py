import time
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from misc import AverageMeter
import matplotlib.pyplot as plt
import os
""" 
    Class that defines the Dice Loss function.
"""
class DiceLoss(nn.Module):
  
    def __init__(self, smooth = 1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = torch.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)
  
    def forward(self, y_pred, y_true):
        return -self.dice_coef(y_pred, y_true)


""" 
    Class that defines the Root Mean Square Loss function.
"""
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


"""
    Class that defines the Cross Entropy Loss Function
"""
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return -torch.mean(torch.sum(y_true*torch.log(F.softmax(y_pred,dim=1)),dim=1))

"""
    Class that defines the Cross Entropy Loss Function
""" 
class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()

    def forward(self, y_pred, y_true, weights):
        y_true = y_true/(y_true.sum(2).sum(2,dtype=torch.float).unsqueeze(-1).unsqueeze(-1))
        y_true[y_true != y_true] = 0.0
        y_true = torch.sum(y_true,dim=1, dtype = torch.float).unsqueeze(1)
        y_true = y_true * weights.to(torch.float)
        old_range = torch.max(y_true) - torch.min(y_true)
        new_range = 100 - 1
        y_true = (((y_true - torch.min(y_true)) * new_range) / old_range) + 1
        return -torch.mean(torch.sum(y_true*torch.log(F.softmax(y_pred,dim=1)),dim=1))
    

""" 
    Functions that trains a net.
"""
def train_net(net, device, loader, optimizer, criterion, batch_size,epoch, isWCE=False):
    net.train()
    train_loss_history_batch = []
    train_loss_history=[]
    train_loss = AverageMeter()
    time_start = time.time()
    for batch_idx, (data, gt, weights) in enumerate(loader):

        # Use GPU or not
        data, gt = data.to(device), gt.to(device)

        # Forward
        predictions = net(data)
        
        # Loss Calculation
        if not isWCE:
            loss = criterion(predictions, gt)
        else:
            weights = weights.to(device)
            loss = criterion(predictions, gt, weights)

        # Updates the record
        train_loss.update(loss.item(), predictions.size(0))
        train_loss_history_batch.append(loss.item())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
            batch_idx * len(data), len(loader)*batch_size,
            100. * batch_idx / len(loader), loss.item()))
    time_dif = time.time() - time_start
    print('\nAverage Training Loss: ' + str(train_loss.avg))
    print('Train Time: It tooks %.4fs to finish the epoch.' % (time_dif))
    fig, ax = plt.subplots()
    batches = np.arange(1, len(train_loss_history_batch) + 1)
    ax.plot(batches, train_loss_history_batch)
    ax.set_xlabel('Batch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Train Loss History batch')
    save_dir_train='plots/train'
    plt.savefig(save_dir_train + '/train_loss_plot_epoch{}.png'.format(epoch))
    # Save the plot with the number of epochs in the filename
    #plt.savefig('train_loss_plot_epoch{}.png'.format(epoch))
    plt.close(fig)        
    return train_loss.avg


""" 
    Function that validates the net.
"""
def val_net(net, device, loader, criterion, batch_size, epoch):
    net.eval()
    val_loss = AverageMeter()
    time_start = time.time()
    val_loss_history_batch = []
    accuracy_history=[]
    with torch.no_grad():
        for batch_idx, (data, gt, weights) in enumerate(loader):

            # Use GPU or not
            data, gt = data.to(device), gt.to(device)

            # Forward
            predictions = net(data)
            
            # Loss Calculation
            loss = criterion(predictions, gt)
            threshold = 0.01  # Adjust the threshold as per your requirement
            accuracy = calculate_accuracy(predictions, gt, threshold)

            # Updates the record
            val_loss.update(loss.item(), predictions.size(0))
            accuracy_history.append(accuracy)
            print('[{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(loader)*batch_size,
                100. * batch_idx / len(loader), loss.item()))
    
    time_dif = time.time() - time_start
    print('\nValidation set: Average loss: '+ str(val_loss.avg))
    print('Validation time: It tooks %.4fs to finish the Validation.' % (time_dif))
    batches = np.arange(1, len(accuracy_history) + 1)
    plt.plot(batches, accuracy_history)
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Curve')
    save_dir_val='plots/validation'
    plt.savefig(save_dir_val + '/val_accuracy_plot_epoch{}.png'.format(epoch))
    plt.close()  # Close the plot
    return val_loss.avg
# Define a function to calculate the accuracy based on pixel closeness
def calculate_accuracy(predictions, gt, threshold):
    # Move tensors to CPU if they are on GPU
    if predictions.device.type == 'cuda':
        predictions = predictions.cpu()
    if gt.device.type == 'cuda':
        gt = gt.cpu()
    
    # Convert tensors to NumPy arrays
    predictions = predictions.detach().numpy()
    gt = gt.detach().numpy()
    
    # Calculate the absolute difference between predictions and ground truth
    diff = np.abs(predictions - gt)
    
    # Count the number of pixels that are close together
    close_pixels = np.count_nonzero(diff <= threshold)
    
    # Calculate the accuracy percentage
    accuracy = (close_pixels / predictions.size) * 100
    
    return accuracy