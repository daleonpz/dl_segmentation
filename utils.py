from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms as T

class CityscapesDownsampled(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, transform=None, target_transform=None):
        self.ignore_index=250
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = torch.load(img_path)
        self.labels = torch.load(label_path)
        self.transform = transform
        self.target_transform = target_transform
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
                    'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
                    'train', 'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.n_classes=len(self.valid_classes)
        self.colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

        self.label_colours = dict(zip(range(self.n_classes), self.colors))
    
    def __len__(self):
        return len(self.imgs)
    
    def encode_segmap(self, mask):
        # remove unwanted classes and recitify the labels of wanted classes
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask.long().squeeze()

    def decode_segmap(self, temp):
        # convert gray scale to color
        temp = temp.numpy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def seg_show(self, seg):
        """
        shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        seg = self.decode_segmap(seg.squeeze_())
        plt.imshow(seg)
    
    def plot_triplet(self, img, seg, pred):
        """
        shows a triplet of: image + ground truth + predicted segmentation
        """
        plt.subplots(ncols=3, figsize=(18,10))
        plt.subplot(131)
        plt.title("Original Image")
        self.img_show(img, mean=torch.tensor([0.5]), std=torch.tensor([0.5]))
        plt.subplot(132)
        plt.title("Ground Truth")
        self.seg_show(seg)
        plt.subplot(133)
        plt.title("Predicted")
        self.seg_show(pred)
        plt.show()
    
    def img_show(self, img, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
        """
        shows an image on the screen.
        mean of 0 and variance of 1 will show the images unchanged in the screen
        """
        unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        npimg = unnormalize(img).numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def __getitem__(self, index):
        img = self.imgs[index,...]
        seg = self.labels[index,...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)
        
        return img, seg
    

class CityscapesSubset(CityscapesDownsampled):
    def __init__(self, dataset, indices, **params):
        super().__init__(**params)
        self.train_dataset_test = torch.utils.data.Subset(dataset, indices)
    def __len__(self):
        return len(self.train_dataset_test)
    
    def __getitem__(self, index):
        img = self.imgs[index,...]
        seg = self.labels[index,...]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            seg = self.target_transform(seg)
        
        return img, seg


def validate(model, val_loader, device, criterion, metric):
    model.eval()
    loss_step, miou_step = [], []
    
    with torch.no_grad():
        for inp_data, labels in val_loader:
            labels = labels.to(device)
            inp_data = inp_data.to(device)
            outputs = model(inp_data)
            val_loss = criterion(outputs, labels)
            miou_step.append(metric(outputs, labels).item())
            loss_step.append(val_loss.item())
        
        val_loss_epoch = np.mean(loss_step)
        val_mIoU = np.mean(miou_step)*100
        return val_loss_epoch, val_mIoU

def train_one_epoch(model, optimizer, train_loader, device, criterion, metric):
    model.train()
    loss_step, miou_step = [], []
    for (inp_data, labels) in train_loader:
        labels = labels.to(device)
        inp_data = inp_data.to(device)
        outputs = model(inp_data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            miou_step.append(metric(outputs, labels).item())
            loss_step.append(loss.item())
    
    loss_curr_epoch = np.mean(loss_step)
    train_mIoU = np.mean(miou_step)*100
    return loss_curr_epoch, train_mIoU


def train(model, optimizer, num_epochs, train_loader, val_loader, device, criterion, metric, exp_name='unet', viz=False, viz_freq=20):
    best_val_metric = -1
    model = model.to(device)
    dict_log = {"train_mIoU":[], "val_mIoU":[], "train_loss":[], "val_loss":[]}
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_loss, train_metric = train_one_epoch(model, optimizer, train_loader, device, criterion, metric)
        val_loss, val_metric,  = validate(model, val_loader, device, criterion, metric)
        msg = (f'Ep {epoch}/{num_epochs}: mIoU : Train:{train_metric:.3f} \t Val:{val_metric:.2f}\
                || Loss: Train {train_loss:.3f} \t Val {val_loss:.3f}')
        
        pbar.set_description(msg)

        dict_log["train_mIoU"].append(train_metric)
        dict_log["val_mIoU"].append(val_metric)
        dict_log["train_loss"].append(train_loss)
        dict_log["val_loss"].append(val_loss)

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  'mIoU':val_metric,
                  }, f'{exp_name}_best_model_min_val_loss.pth')
        if viz and (epoch+1)%viz_freq==0:
            show_preds(model, train_loader, device, num_samples=1)
     
    return dict_log

def show_preds(model, loader, device, ignore_index=250, num_samples=1 ):
    model.eval()
    model = model.to(device)
    imgs, segs = next(iter(loader))
    preds = model(imgs.to(device))
    num_samples = min(num_samples, int(imgs.shape[0]))
    for img_id in range(num_samples):
        pred = preds.argmax(dim=1)[img_id,...].cpu()
        img, seg = imgs[img_id,...], segs[img_id,...]
        pred[seg==ignore_index] = ignore_index
        if hasattr(loader.dataset, 'plot_triplet'):
            loader.dataset.plot_triplet(img, seg, pred)
        elif hasattr(loader.dataset.dataset, 'plplot_tripletot'):
            loader.dataset.dataset.plot_triplet(img, seg, pred)
        else:
            raise NotImplementedError("Dataset does not have plot_triplet method")

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

def plot_stats(dict_log, modelname="",baseline=None, title=None, scale_metric=100):
    plt.figure(figsize=(15,10))
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1)
    x_axis = list(range(len(dict_log["val_metric"])))
    
    y_axis_train = [i * scale_metric for i in dict_log["train_metric"]]
    y_axis_val = [i * scale_metric for i in dict_log["val_metric"]]
    plt.plot(y_axis_train, label=f'{modelname} Train mIoU')
    plt.scatter(x_axis, y_axis_train)

    plt.plot( y_axis_val, label=f'{modelname} Validation mIoU')
    plt.scatter(x_axis, y_axis_val)

    plt.ylabel('mIoU in %')
    plt.xlabel('Number of Epochs')
    plt.title("mIoU over epochs", fontsize=fontsize)
    if baseline is not None:
        plt.axhline(y=baseline, color='red', label="Acceptable performance")
    plt.legend(fontsize=fontsize, loc='best')

    plt.subplot(2,1,2)
    plt.plot(dict_log["train_loss"] , label="Training")


    plt.scatter(x_axis, dict_log["train_loss"], )
    plt.plot(dict_log["val_loss"] , label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper right')
    if title is not None:
        plt.savefig(title, bbox_inches='tight', dpi=400)

