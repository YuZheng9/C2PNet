import os
import os.path as osp
import pickle
import random
import sys

import lmdb
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FF

from metrics import *
from option import opt

sys.path.append('net')
sys.path.append('')
BS = opt.bs
print(BS)
crop_size = 'whole_img'
path = opt.dataset_dir
if opt.crop:
    crop_size = opt.crop_size


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

    def __getitem__(self, index):
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('/')[-1].split('_')[0]
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        if opt.norm:
            data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


class LMDB_Image:
    def __init__(self, haze, clear, n1, n2, n3, n4, n5, n6):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = haze.shape[2]
        self.size = haze.shape[:2]

        self.haze = haze.tobytes()
        self.clear = clear.tobytes()
        self.n1 = n1.tobytes()
        self.n2 = n2.tobytes()
        self.n3 = n3.tobytes()
        self.n4 = n4.tobytes()
        self.n5 = n5.tobytes()
        self.n6 = n6.tobytes()

    def get_image(self):
        """ Returns the image as a numpy array. """
        haze = np.frombuffer(self.haze, dtype=np.uint8)
        clear = np.frombuffer(self.clear, dtype=np.uint8)
        n1 = np.frombuffer(self.n1, dtype=np.uint8)
        n2 = np.frombuffer(self.n2, dtype=np.uint8)
        n3 = np.frombuffer(self.n3, dtype=np.uint8)
        n4 = np.frombuffer(self.n4, dtype=np.uint8)
        n5 = np.frombuffer(self.n5, dtype=np.uint8)
        n6 = np.frombuffer(self.n6, dtype=np.uint8)

        return haze.reshape(*self.size, self.channels), clear.reshape(*self.size, self.channels), n1.reshape(*self.size,
                                                                                                             self.channels), n2.reshape(
            *self.size, self.channels), n3.reshape(*self.size, self.channels), n4.reshape(*self.size,
                                                                                          self.channels), n5.reshape(
            *self.size, self.channels), n6.reshape(*self.size, self.channels)


class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, size):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.size = size
        print('crop size', size)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
            # self.NP= pickle.loads(txn.get(b'__NP__'))
        # self.class_weights=torch.load("/data/jxzhang/coco/data/classweight.pt")

    def aug_data(self, *images):
        out = []
        for img in images:
            out.append(tfs.ToTensor()(img))
        if not isinstance(self.size, str):

            i, j, h, w = tfs.RandomCrop.get_params(out[0], output_size=(self.size, self.size))
            for idx, img in enumerate(out):
                out[idx] = FF.crop(out[idx], i, j, h, w)
        rand_hor = random.randint(0, 1)
        rand_rot = random.randint(0, 3)
        for idx, img in enumerate(out):
            out[idx] = tfs.RandomHorizontalFlip(rand_hor)(out[idx])
            if rand_rot:
                out[idx] = FF.rotate(out[idx], 90 * rand_rot)
        out.append(out[0])
        if opt.norm:
            out[0] = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(out[0])
        return out

    def __getitem__(self, index):
        env = self.env
        with env.begin() as txn:
            byteflow = txn.get(self.keys[index])
        IMAGE = pickle.loads(byteflow)
        haze, clear, n1, n2, n3, n4, n5, n6 = IMAGE.get_image()
        if isinstance(self.size, int):
            while haze.shape[0] < self.size or haze.shape[1] < self.size:
                index = random.randint(0, 300000)
                with env.begin() as txn:
                    byteflow = txn.get(self.keys[index])
                IMAGE = pickle.loads(byteflow)
                haze, clear, n1, n2, n3, n4, n5, n6 = IMAGE.get_image()
        out = self.aug_data(haze, clear, n1, n2, n3, n4, n5, n6)
        return out

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


ITS_train_loader_lmdb = DataLoader(
    dataset=DatasetLMDB(os.path.join(path, 'ITS/ITS.lmdb'), size=crop_size), batch_size=BS,
    shuffle=True, pin_memory=True)
ITS_test_loader = DataLoader(dataset=RESIDE_Dataset(os.path.join(path, 'SOTS/indoor'), train=False, size='whole img'),
                             batch_size=1, shuffle=False)
# OTS_train_loader_all = DataLoader(
#     dataset=DatasetLMDB(os.path.join(path, 'OTS/OTS.lmdb'), size=crop_size), batch_size=BS,
#     shuffle=True, pin_memory=True)
OTS_test_loader = DataLoader(
    dataset=RESIDE_Dataset(os.path.join(path, 'SOTS/outdoor'), train=False, size='whole img', format='.png'),
    batch_size=1,
    shuffle=False)
