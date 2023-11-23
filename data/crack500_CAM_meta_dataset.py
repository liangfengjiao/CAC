# Author: Yahui Liu <yahui.liu@unitn.it>

import os.path
import random
import cv2
import numpy as np
from PIL import Image
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from data.image_folder import make_dataset
from data.utils import MaskToTensor, get_params, affine_transform


class Crack500CAMMetaDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, opt, confidence, mask):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.img_paths = make_dataset(os.path.join(opt.dataroot, '{}'.format(opt.phase), 'image'))

        if opt.phase == 'train':
            datapath = os.path.join(opt.dataroot, '{}'.format(opt.phase), opt.pseudo_label, opt.proportion_c)
        else:
            datapath = os.path.join(opt.dataroot, '{}'.format(opt.phase))
        self.lab_dir = os.path.join(datapath, confidence)

        self.mask_dir = os.path.join(datapath, mask)

        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        self.lab_transform = MaskToTensor()

        self.mask_transform = MaskToTensor()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path).split('.')[0] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # adjust the image size
        w, h = self.opt.load_width, self.opt.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)

        # binarize segmentation
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # apply flip
        if (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                img = np.fliplr(img)
                lab = np.fliplr(lab)
                mask = np.fliplr(mask)
            else:
                img = np.flipud(img)
                lab = np.flipud(lab)
                mask = np.flipud(mask)

        # apply affine transform
        if self.opt.use_augment:
            if random.random() > 0.5:
                angle, scale, shift = get_params()
                img = affine_transform(img, angle, scale, shift, w, h)
                lab = affine_transform(lab, angle, scale, shift, w, h)
                mask = affine_transform(mask, angle, scale, shift, w, h)

        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # apply the transform to both A and B
        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)
        mask = self.lab_transform(mask.copy()).unsqueeze(0)

        return {'image': img, 'label': lab, 'mask': mask, 'A_paths': img_path, 'B_paths': lab_path, 'mask_dir': mask_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
