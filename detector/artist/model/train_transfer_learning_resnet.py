## ALL MAIN ARGUMENTS FOR THE SCRIPT ##
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from painting_loader import PaintingFolder

import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T

import torchnet as tnt
from torchnet.meter import ConfusionMeter

import numpy as np
import timeit

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import cv2
import os, glob
import numpy as np
import warnings

def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def train(model, loss_fn, optimizer, loader_train, loader_val, train_acc, val_acc, num_epochs = 1, dtype = torch.cuda.FloatTensor):
    train_loss_hist = []
    print_every = 100
    # val_loss_hist = []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # record training loss history
        train_loss_hist.append(loss)
            
        # record training and validation accuracy at the end of each epoch
        train_acc.append(check_accuracy(model, loader_train))
        val_acc.append(check_accuracy(model, loader_val))

        torch.save(model.state_dict(), '../artist_ckp/state_dict.dat.' + str(val_acc[-1]))
        torch.save([train_acc, val_acc], '../artist_ckp/train_val_accs.dat.' + str(val_acc[-1]))
        
    return [train_acc, val_acc, train_loss_hist]
    
def check_accuracy(model, loader, dtype = torch.cuda.FloatTensor, ytype = torch.LongTensor):
    print('Checking accuracy!')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.type(ytype)
        x_var = Variable(x.type(dtype))
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        torch.save([preds, y], '../artist_ckp/preds_y.dat')
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    print(num_correct, num_samples)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100*acc

def check_accuracy_topX(model,loader, top=5, dtype = torch.cuda.FloatTensor, ytype = torch.LongTensor):
    print('Checking top' + str(top) + ' accuracy!')
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.view(-1, 1).type(ytype)
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)
        
        s = scores.data.cpu().numpy()
        ind = np.argpartition(s, -top)[:, -top:]

        # crappy loop... must be a vectorized way to do this
        c = 0
        y_n = y.numpy()
        for i in np.arange(ind.shape[0]):
            if y[i,0] in ind[i]:
                c += 1
        
        num_correct += c
        num_samples += ind.shape[0]
        
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100*acc    

def confusion_matrix(model, loader, conf, dtype = torch.cuda.FloatTensor, ytype = torch.LongTensor):
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.view(-1, 1).type(ytype)
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)
        
        conf.add(scores.data, y)

def filter_split_dataset(dataset_folder, filter_subset = False, num_train = 100, num_val = 30, num_test = 30, num_samples = 300, seed = 231):
    t = pd.read_csv(dataset_folder + 'all_data_info.csv')
    # filter down (if needed)
    if (filter_subset):
        t = t[t['new_filename'].str.startswith('1')]
        t = t[t['in_train']]
    
    t.head()
    print(t.shape)
    
    # list of all artists to include
    temp = t['artist'].value_counts()
    threshold = num_samples
    # threshold = 500
    artists = temp[temp >= threshold].index.tolist()
    artists.sort()
    artists = artists[:4] + ["Vincent van Gogh"]
    # output the artists we are using to artists_class.txt
    with open('artists_class.txt', 'w') as f:
        for item in artists:
            f.write("%s\n" % item)
    num_artists = len(artists)
    print(str(len(artists)) + ' artists being classified')
    
    # pull train and val data for just those artists
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for a in artists:
        df = t[t['artist'].str.startswith(a, na=False)]#.sample(n=num_samples, random_state=seed)
        t_df = df  # Use all the data for training
        # Use all the date for validation
        v_df = t_df # .sample(n=num_val, random_state=seed)
        te_df = t_df # We do not check test accuracy

        train_dfs.append(t_df)
        val_dfs.append(v_df)
        test_dfs.append(te_df)
    
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)
    print("Done")
    return [train_df, val_df, test_df, num_artists]

def create_data_loader(train_df, val_df, test_df, b_size = 60, num_workers = 4, img_folder = '../../../../../data/image/'):
    mean_resnet = np.array([0.485, 0.456, 0.406])
    std_resnet = np.array([0.229, 0.224, 0.225])
            
    train_transform = T.Compose([
            T.Resize([224,224]),
            #T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean_resnet, std_resnet)
        ])
    val_transform = T.Compose([
            #T.Scale(256),
            T.Resize([224,224]),
            T.ToTensor(),
            T.Normalize(mean_resnet, std_resnet)
        ])
    
    train_dset = PaintingFolder(img_folder, train_transform, train_df)
    loader_train = DataLoader(train_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        
    val_dset = PaintingFolder(img_folder, val_transform, val_df)
    loader_val = DataLoader(val_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
    
    test_dset = PaintingFolder(img_folder, val_transform, test_df)
    loader_test = DataLoader(test_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
    return loader_train, loader_val, loader_test

def create_model(num_artists, device = 'cuda:0'):
    import torchvision
    # transfer learning on top of ResNet (only replacing final FC layer)
    # model_conv = torchvision.models.resnet18(pretrained=True)
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_artists)
    
    if torch.cuda.is_available():
        model_conv = model_conv.to(torch.device(device))
    return model_conv

def train_model(loader_train, loader_val, loader_test, num_artists, dtype = torch.cuda.FloatTensor):
    fc_epoch = 10
    model_epoch = 3
    model_conv = create_model(num_artists)
    loss_fn = nn.CrossEntropyLoss().type(dtype)
    
    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=1e-3)
    
    train_acc = []
    val_acc = []
    # Load last training
    if os.path.isfile('../artist_ckp/state_dict.dat'):
        print("Loading state files...") 
        model_conv.load_state_dict(torch.load('../artist_ckp/state_dict.dat'))
        import shutil
        shutil.copyfile('../artist_ckp/state_dict.dat', '../artist_ckp/state_dict.dat.bk')
    train_acc, val_acc = [], []
    
    print("Back up state files into .bk files...")
    
    start_time = timeit.default_timer()
    train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = fc_epoch)
    
    print()
    print(str(timeit.default_timer() - start_time) + " seconds taken")
    
    # now we allow all of the network to change, but by less
    for param in model_conv.parameters():
        param.requires_grad = True
    
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=1e-4, weight_decay=1e-2)
    
    start_time = timeit.default_timer()
    train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = model_epoch)
    
    print(str(timeit.default_timer() - start_time) + " seconds taken")
    
    #check_accuracy(model_conv, loader_test)
    
    torch.save(model_conv.state_dict(), '../artist_ckp/state_dict.dat')
    torch.save([train_acc, val_acc], '../artist_ckp/train_val_accs.dat')

def main():
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    dtype = torch.FloatTensor
    ytype = torch.LongTensor
    ytype_cuda = torch.cuda.LongTensor
    if (torch.cuda.is_available()):
        dtype = torch.cuda.FloatTensor
    print(ytype)
    print(dtype)
    # ================ TRAINING CONFIG ==================
    seed = 231
    random.seed(seed)
    dat_folder = '../../../../../data/'
    img_folder = dat_folder + 'image/'
    num_workers = 4

    filter_subset = False # True if we want to filter to just train _1
    balanced_dset = True # True if I want equal # of paintings per artist, false if I want to use all available per artist

    ## THIS VERSION OF SCRIPT HAS EQUAL NUMBER OF PAINTINGS PER ARTIST
    num_train = 240
    num_val = 30
    num_test = num_val
    num_samples = num_train + num_val + num_test # threshold to include an artist
    b_size = 60 # batch size for the data loaders
    
    train_df, val_df, test_df, num_artists = filter_split_dataset(dat_folder)
    loader_train, loader_val, loader_test = create_data_loader(train_df, val_df, test_df)
    print("Data Loader Creation Done")
    train_model(loader_train, loader_val, loader_test, num_artists)

if __name__ == '__main__':
    main()