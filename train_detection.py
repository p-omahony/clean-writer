from dataset.dataset_factory import FormDataset
from models.detection import create_backbone

import torchvision
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import torch

import pandas as pd
from tqdm import tqdm

def collate_fn(batch) :
    imgs=[]
    labels=[]
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    return imgs, labels

def moveTo(obj, device) :
      if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
      elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
      elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
      elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
              to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
      elif hasattr(obj, 'to'):
        return obj.to(device)
      else:
        return obj

labels = pd.read_csv('./data/lines.txt', sep=' ', comment='#', usecols=range(8), header=None)
labels.columns = ['name', 'status', 'graylevel', 'nc', 'xmin', 'ymin', 'w', 'h']
labels = labels[['name', 'xmin', 'ymin', 'w', 'h']]
labels['name'] = labels['name'].apply(lambda x: '-'.join(x.split('-')[:-1]))
labels['xmax'] = labels['xmin'] + labels['w']
labels['ymax'] = labels['ymin'] + labels['h']
labels = labels[['name', 'xmin', 'ymin', 'xmax', 'ymax']]
labels = labels.groupby('name', as_index=False).agg(list)
labels = labels[labels['name'].str[0].isin(['a', 'b', "c", "d"])].reset_index(drop=True)

dataset = FormDataset('./data', labels, transform=ToTensor(), target_transform=ToTensor())
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

backbone = create_backbone()
anchor_generator = AnchorGenerator(sizes=((32),), aspect_ratios=((1.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

model = FasterRCNN(backbone, num_classes=1, image_mean=[0.5], image_std=[0.229], min_size=100, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.train()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters())

for epoch in tqdm(range(5), desc='Epoch', disable=False):
    running_loss = 0.0
    final_losses, iters = [], []
    iter_ = 0
    for inputs, labels in tqdm(train_loader, desc='Train Batch', leave=True, position=0, disable=False):
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        optimizer.zero_grad()
        losses = model(inputs, labels)
        loss = 0
        for partial_loss in losses.values():
            loss += partial_loss
        loss.backward()
        final_losses.append(loss)
        iters.append(iter_)
        iter_+=1

        optimizer.step()

    running_loss += loss.item()
    torch.save('model_%s'%epoch+'.pt')