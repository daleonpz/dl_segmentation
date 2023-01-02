from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt


def validate(model, val_loader, device, criterion, metric):
    loss_step, metric_step = [], []
    ### START CODE HERE ### (approx. 11 lines)
    model.eval()
    with torch.no_grad():
        for inp_data, labels in val_loader:
            # Move the data to the GPU
            labels = labels.view(labels.shape[0]).to(device)
            inp_data = inp_data.to(device)
            outputs = model(inp_data)
            val_loss = criterion(outputs, labels)
            metric_step.append(metric(outputs, labels))
            loss_step.append(val_loss.item())

    # dont forget to take the means here
    val_loss_epoch = torch.tensor(loss_step).mean().numpy()
    metric_epoch = torch.tensor(metric_step).mean().numpy()
    ### END CODE HERE ###
    return val_loss_epoch, metric_epoch


def train_one_epoch(model, optimizer, train_loader, device, criterion, metric):
    loss_step, metric_step = [], []
    ### START CODE HERE ### (approx. 11 lines)
    model.train()
    for inp_data, labels in train_loader:
        # Move the data to the GPU
        labels = labels.view(labels.shape[0]).to(device)
        inp_data = inp_data.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_step.append(metric(outputs, labels))
        loss_step.append(loss.item())
    # dont forget the means here
    loss_curr_epoch = torch.tensor(loss_step).mean().numpy()
    metric_epoch = torch.tensor(metric_step).mean().numpy()
    ### END CODE HERE ###
    return loss_curr_epoch, metric_epoch

def show_preds(model, loader, device, ignore_index=250, num_samples=1):
    ### START CODE HERE ### (approx. 11 lines)
    model.eval()

    # get one sample from the loader
    for inp_data, labels in loader:
        # Move the data to the GPU
        labels = labels.view(labels.shape[0]).to(device)
        inp_data = inp_data.to(device)
        outputs = model(inp_data)
        break
            
    print(labels)
    ### END CODE HERE ###
    # Visualizes the three arguments in a plot
    loader.dataset.plot_triplet(img, seg, pred)

def train(model, optimizer, num_epochs, train_loader, val_loader, device, criterion, metric, exp_name='unet', viz=False, viz_freq=20):
    best_val_metric = -1
    model = model.to(device)
    dict_log = {"train_metric":[], "val_metric":[], "train_loss":[], "val_loss":[]}
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        ### START CODE HERE ### (approx. 2 lines)
        train_loss, train_metric    = train_one_epoch(model, optimizer, train_loader, device, criterion, metric)
        val_loss, val_metric        = validate(model, val_loader, device, criterion, metric)
        ### END CODE HERE ###

        msg = (f'Ep {epoch}/{num_epochs}: metric : Train:{train_metric:.3f} \t Val:{val_metric:.2f}\
                || Loss: Train {train_loss:.3f} \t Val {val_loss:.3f}')

        pbar.set_description(msg)

        dict_log["train_metric"].append(train_metric)
        dict_log["val_metric"].append(val_metric)
        dict_log["train_loss"].append(train_loss)
        dict_log["val_loss"].append(val_loss)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  'metric':val_metric,
                  }, f'{exp_name}_best_model_min_val_loss.pth')

        if viz and (epoch+1) % viz_freq==0:
            show_preds(model, train_loader, device, num_samples=1)

    return dict_log


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

def test_model(model, path, test_loader, device='cuda'):
    model = load_model(model, path)
    model.to("cuda")
    model.eval()
    return validate(model, test_loader, device)


def plot_stats(dict_log, modelname="",baseline=90, title=None):
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1)
    ### START CODE HERE ### (approx. 5 lines)
    x_axis = list(range(len(dict_log["val_acc_epoch"])))
    plt.plot(dict_log["train_acc_epoch"], label=f'{modelname} Train accuracy')
    plt.scatter(x_axis, dict_log["train_acc_epoch"])

    plt.plot(dict_log["val_acc_epoch"], label=f'{modelname} Validation accuracy')
    plt.scatter(x_axis, dict_log["val_acc_epoch"])
    ### END CODE HERE ###


    plt.ylabel('Accuracy in %')
    plt.xlabel('Number of Epochs')
    plt.title("Accuracy over epochs", fontsize=fontsize)
    plt.axhline(y=baseline, color='red', label="Acceptable accuracy")
    plt.legend(fontsize=fontsize)


    plt.subplot(2,1,2)
    plt.plot(dict_log["loss_epoch"] , label="Training")

    ### START CODE HERE ### (approx. 3 lines)
    plt.scatter(x_axis, dict_log["loss_epoch"], )
    plt.plot(dict_log["val_loss"] , label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])
    ### END CODE HERE ###

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if title is not None:
        plt.savefig(title)
