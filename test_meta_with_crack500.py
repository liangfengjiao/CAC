from options.test_options import TestOptions
import os
import torch
import numpy as np
import cv2
from data import create_dataset
from models.deepcrack_meta_networks import define_meta_deepcrack

def main(opt):
    #dataset_input
    epoch_itrs = '40_Meta_crack500'
    result_path = os.path.join(opt.results_dir, 'CrackForest', opt.name, epoch_itrs, 'test_latest', 'images')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    opt.batch_size = 1
    #opt.phase = 'train'
    dataset_image = create_dataset(opt, 'mask', 'mask')
    dataset_iter = enumerate(dataset_image)

    #load crack segmentation model
    meta_crackSegNet = define_meta_deepcrack(opt.input_nc,
                                             opt.num_classes,
                                             opt.ngf,
                                             opt.norm,
                                             opt.init_type,
                                             opt.init_gain,
                                             opt.gpu_ids)
    meta_crackSegNet.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, epoch_itrs+'.pth')))
    meta_crackSegNet.eval()

    #output segmentation result
    for i, data in dataset_iter:
        image_path = data['A_paths'][0].split('/')[-1]
        image_name = image_path.split('.')[0]
        image_postfix = '.png'

        pre = meta_crackSegNet(data)
        fused = np.squeeze(torch.sigmoid(pre[-1]).cpu().detach().numpy())
        side1 = np.squeeze(torch.sigmoid(pre[0]).cpu().detach().numpy())
        side2 = np.squeeze(torch.sigmoid(pre[1]).cpu().detach().numpy())
        side3 = np.squeeze(torch.sigmoid(pre[2]).cpu().detach().numpy())
        side4 = np.squeeze(torch.sigmoid(pre[3]).cpu().detach().numpy())
        side5 = np.squeeze(torch.sigmoid(pre[4]).cpu().detach().numpy())

        src_img = cv2.imread(data['A_paths'][0], 0)
        label_img = cv2.imread(data['B_paths'][0], 0)
        fused = cv2.resize(fused * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side1 = cv2.resize(side1 * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side2 = cv2.resize(side2 * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side3 = cv2.resize(side3 * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side4 = cv2.resize(side4 * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        side5 = cv2.resize(side5 * 255, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        src_name = image_name + '_image' + image_postfix
        label_name = image_name + '_label_viz' + image_postfix
        fused_name = image_name + '_fused' + image_postfix
        side1_name = image_name + '_side1' + image_postfix
        side2_name = image_name + '_side2' + image_postfix
        side3_name = image_name + '_side3' + image_postfix
        side4_name = image_name + '_side4' + image_postfix
        side5_name = image_name + '_side5' + image_postfix

        cv2.imwrite(os.path.join(result_path, src_name), src_img)
        cv2.imwrite(os.path.join(result_path, label_name), label_img)
        cv2.imwrite(os.path.join(result_path, fused_name), fused)
        cv2.imwrite(os.path.join(result_path, side1_name), side1)
        cv2.imwrite(os.path.join(result_path, side2_name), side2)
        cv2.imwrite(os.path.join(result_path, side3_name), side3)
        cv2.imwrite(os.path.join(result_path, side4_name), side4)
        cv2.imwrite(os.path.join(result_path, side5_name), side5)








if __name__ == '__main__':
    opt = TestOptions().parse()
    main(opt)