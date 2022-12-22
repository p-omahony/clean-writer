import os
import numpy as np
import pickle
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from skimage.color import rgb2gray
from skimage.transform import rotate
import matplotlib as mpl
import matplotlib.pyplot as plt
from ds_ctcdecoder import Alphabet, ctc_beam_search_decoder, Scorer

from dataset.preprocessing import preprocess_img, read_labels
from dataset.dataset_factory import IAMData
from models.model import IAMModel

def collate(batch):
    images, words = [b[0] for b in batch], [b[1] for b in batch]
    images = torch.stack(images, 0)
    # Calculate target lengths for the current batch
    lengths = [len(word) for word in words]
    # According to https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
    # Tensor of size sum(target_lengths) the targets are assumed to be un-padded and concatenated within 1 dimension.
    targets = torch.empty(sum(lengths)).fill_(len(classes)).long()
    lengths = torch.tensor(lengths)
    # Now we need to fill targets according to calculated lengths
    for j, word in enumerate(words):
        start = sum(lengths[:j])
        end = lengths[j]
        targets[start:start + end] = torch.tensor([char_dict.get(letter) for letter in word]).long()
    return images.to(dev), targets.to(dev), lengths.to(dev)


np.random.seed(42)
torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = read_labels('./data/lines.txt')
df = df.head(1)
dataset = IAMData(df)


# f = open('chars.txt', 'r')
# lines = f.readlines()
# classes = ''.join(lines).replace('\n', '')
# print(classes)

with open('char_dict.pkl', 'rb') as f:
    char_dict = pickle.load(f)

classes = ''.join(char_dict.keys())
text_file = open("chars.txt", "w", encoding='utf-8')
text_file.write('\n'.join([x if x != '#' else '\\#' for x in char_dict.keys()]))
text_file.close()


model = IAMModel(time_step=96,
                feature_size=512,
                hidden_size=512,
                output_size=len(classes) + 1,
                num_rnn_layers=4)
model.load_state_dict(torch.load('./weights/7_222176942825706_model.pth', map_location=torch.device('cpu')))
model.to(dev)

validation_batch_size = 1
decode_map = {v: k for k, v in char_dict.items()}

validation_loader = DataLoader(dataset, batch_size=validation_batch_size, collate_fn=collate)

def batch_predict(model, valid_dl, up_to):
    val_iter = iter(valid_dl)
    xb, yb, lens = next(val_iter)
    model.eval()
    with torch.no_grad():
        outs = model.beam_search_with_lm(xb)
        for i in range(len(outs)):
            start = sum(lens[:i])
            end = lens[i].item()
            corr = ''.join([decode_map.get(letter.item()) for letter in yb[start:start + end]])
            predicted = ''.join([letter for letter in outs[i]])
            # ============================================ SHOW IMAGE ==================================================
            img = xb[i, :, :, :].permute(1, 2, 0).cpu().numpy()
            img = rgb2gray(img)
            img = rotate(img, angle=90, clip=False, resize=True)
            f, ax = plt.subplots(1, 1)
            mpl.rcParams["font.size"] = 8
            ax.imshow(img, cmap='gray')
            mpl.rcParams["font.size"] = 14
            plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(corr))
            plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(predicted))
            f.set_size_inches(10, 3)
            print('actual: {}'.format(corr))
            print('predicted:   {}'.format(predicted))
            if i + 1 == up_to:
                break
    plt.show()


batch_predict(model=model, valid_dl=validation_loader, up_to=1)