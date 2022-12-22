from .preprocessing import get_mapping, preprocess_img

from torch.utils.data import Dataset


class IAMData(Dataset):
    def __init__(self, df):
        self.df = df
        self.char_dict = get_mapping(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, word = self.df.iloc[idx, 0], self.df.iloc[idx, 1]
        image, word = preprocess_img(img_name, word)

        return image, word