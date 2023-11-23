from options.train_options import TrainOptions
from data import create_dataset
# from models import create_model
# from util.visualizer import Visualizer
from models.deepcrack_meta_networks import define_meta_deepcrack, BinaryFocalLoss, DMI_Loss
import torch
import time
import os
import cv2
import numpy as np
from division_confidence_datasets import cal_threshold
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def updata_train_datset(opt, model, epoch):
    opt.batch_size = 1
    layer = opt.layer#0:side1, 1:side2, 2:side3, 3:side4, 4:side5, 5(-1):fused;
    dataset_image = create_dataset(opt, 'mask_CAM_proportion_confidence_high', 'mask_CAM_proportion_confidence_low')
    dataset_iter = enumerate(dataset_image)
    dataset_path = os.path.join(opt.dataroot, 'train', opt.pseudo_label, opt.proportion_c)
    dataset_save_path = os.path.join(dataset_path, opt.name, str(epoch))

    pre_path = os.path.join(dataset_path, 'mask_CAM')
    confidence_high_path = os.path.join(dataset_path, 'mask_CAM_proportion_confidence_high')
    confidence_low_path = os.path.join(dataset_path, 'mask_CAM_proportion_confidence_low')
    pre_save_path = os.path.join(dataset_save_path, 'mask_CAM')
    confidence_save_high_path = os.path.join(dataset_save_path, 'confidence_high')
    confidence_save_low_path = os.path.join(dataset_save_path, 'confidence_low')
    if not os.path.exists(pre_save_path):
        os.makedirs(pre_save_path)
    if not os.path.exists(confidence_save_high_path):
        os.makedirs(confidence_save_high_path)
    if not os.path.exists(confidence_save_low_path):
        os.makedirs(confidence_save_low_path)
    for _, data in dataset_iter:
        image_path = data['A_paths'][0].split('/')[-1]
        src_img = cv2.imread(data['A_paths'][0], 0)
        pre = model(data)
        fused = torch.sigmoid(pre[layer])
        fused = np.squeeze(fused.cpu().detach().numpy())
        fused = cv2.resize(fused, (src_img.shape[1], src_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # using side1 or fused-side to reweight pseudo label
        fused_pre = cv2.imread(os.path.join(pre_path, os.path.splitext(image_path)[0] + ".png"), 0)
        #fused = (fused_pre * 0.5 + fused * 0.5 * 255).astype(np.uint8)  #a=0.5
        fused = torch.mul(torch.as_tensor(fused_pre), torch.as_tensor(fused)).numpy()
        if fused.max() - fused.min() > 0:
            fused = (fused - fused.min()) / (fused.max() - fused.min()) * 255
        #fused *= 255
        _, pre_fused = cv2.threshold(fused, thresh=cal_threshold(fused, 0.15), maxval=255, type=cv2.THRESH_BINARY)

        # dividing merge_cam into confidence_high and confidence_low
        _, pre_fused_confidence_high = cv2.threshold(fused, thresh=cal_threshold(fused, 0.05), maxval=255,
                                                     type=cv2.THRESH_BINARY)
        pre_fused_confidence_low = pre_fused - pre_fused_confidence_high


        cv2.imwrite(
            os.path.join(pre_path, os.path.splitext(image_path)[0] + ".png"),
            fused
        )
        cv2.imwrite(
            os.path.join(pre_save_path, os.path.splitext(image_path)[0] + ".png"),
            fused
        )
        cv2.imwrite(
            os.path.join(confidence_high_path, os.path.splitext(image_path)[0] + ".png"),
            pre_fused_confidence_high
        )
        cv2.imwrite(
            os.path.join(confidence_save_high_path, os.path.splitext(image_path)[0] + ".png"),
            pre_fused_confidence_high
        )
        cv2.imwrite(
            os.path.join(confidence_low_path, os.path.splitext(image_path)[0] + ".png"),
            pre_fused_confidence_low
        )
        cv2.imwrite(
            os.path.join(confidence_save_low_path, os.path.splitext(image_path)[0] + ".png"),
            pre_fused_confidence_low
        )




def main(opt):
    # Create model
    meta_train_model = define_meta_deepcrack(opt.input_nc,
                                             opt.num_classes,
                                             opt.ngf,
                                             opt.norm,
                                             opt.init_type,
                                             opt.init_gain,
                                             opt.gpu_ids)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    optimizer_meta_train = torch.optim.SGD(meta_train_model.module.params(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_meta_train, step_size=10, gamma=0.1, verbose=True)
    meta_train_model.train()

    # selected loss function
    if opt.loss_mode == 'DMI':
        criterionSeg = DMI_Loss()
    elif opt.loss_mode == 'focal':
        criterionSeg = BinaryFocalLoss()
    else:
        criterionSeg = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True,
                                                  pos_weight=torch.tensor(1.0 / 3e-2).to(device))  # check

    for epoch in range(opt.epoch_count,
                       opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        opt.batch_size = 4
        total_iters = 0

        dataset_confidence_high = create_dataset(opt, 'mask_CAM_proportion_confidence_high',
                                                 'mask_CAM_proportion_confidence_low')
        dataset_confidence_low = create_dataset(opt, 'mask_CAM_proportion_confidence_low',
                                                'mask_CAM_proportion_confidence_high')

        dataset_confidence_high_iter = enumerate(dataset_confidence_high)
        dataset_confidence_low_iter = enumerate(dataset_confidence_low)

        for i in range(int(len(dataset_confidence_high) / opt.batch_size)):
            total_iters += opt.batch_size
            # meta-train
            _, meta_train_input = dataset_confidence_high_iter.__next__()
            meta_train_label = meta_train_input['label'].cuda()
            meta_train_mask = meta_train_input['mask'].cuda()
            meta_train_pre = meta_train_model(meta_train_input)

            weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]
            lambda_side = opt.lambda_side
            lambda_fused = opt.lambda_fused

            loss_side = 0.0
            for out, w in zip(meta_train_pre[:-1], weight_side):
                # print(meta_train_label)
                loss_side += criterionSeg(out, meta_train_label, meta_train_mask) * w

            # self.loss_fused = self.criterionSeg(self.outputs[-1], self.label3d)
            loss_fused = criterionSeg(meta_train_pre[-1], meta_train_label, meta_train_mask)
            loss_onestep = loss_side * lambda_side + loss_fused * lambda_fused

            # meta-test
            meta_test_model = define_meta_deepcrack(opt.input_nc,
                                                    opt.num_classes,
                                                    opt.ngf,
                                                    opt.norm,
                                                    opt.init_type,
                                                    opt.init_gain,
                                                    opt.gpu_ids)
            meta_test_model.load_state_dict(meta_train_model.state_dict())
            meta_test_model.train()

            # meta_train_model.zero_grad()
            # first-order grad
            grad_info = torch.autograd.grad(
                loss_onestep, meta_train_model.module.params(), create_graph=True
            )

            meta_test_model.module.update_params(
                lr_inner=opt.lr, source_params=grad_info
            )
            _, meta_test_input = dataset_confidence_low_iter.__next__()

            meta_test_label = meta_test_input['label'].cuda()
            meta_test_mask = meta_test_input['mask'].cuda()
            meta_test_pre = meta_test_model(meta_test_input)
            loss_meta_test_side = 0.0
            for out, w in zip(meta_test_pre[:-1], weight_side):
                loss_meta_test_side += criterionSeg(out, meta_test_label, meta_test_mask) * w

            loss_meta_test_fused = criterionSeg(meta_train_pre[-1], meta_train_label, meta_test_mask)
            loss_meta_test = loss_meta_test_side * lambda_side + loss_meta_test_fused * lambda_fused

            loss_meta = (opt.lamda) * loss_onestep + (1 - opt.lamda) * loss_meta_test  # lamda=0.7
            optimizer_meta_train.zero_grad()  # check
            # to check: grad = torch.autograd.grad(loss_meta, self.encoder.module.params())
            # and grad should not be 'None'
            loss_meta.backward()
            optimizer_meta_train.step()

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d, loss=%5f)' % (epoch, total_iters, loss_meta))

        lr_scheduler.step()
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            checkpoints_path = os.path.join(opt.checkpoints_dir, opt.name)
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            torch.save(meta_train_model.state_dict(), os.path.join(checkpoints_path, str(epoch) + '_Meta_crack500.pth'))
        updata_train_datset(opt, meta_train_model, epoch)


if __name__ == '__main__':
    # setting training hyper-parameter
    opt = TrainOptions().parse()
    main(opt)
    # proportion_c_tmp = 0.5
    # for i in range(2):
    #     opt.lamda = lamda_tmp
    #     opt.name = 'crack500_CAM_Location_proportion_c=0.05_lamda='+str(lamda_tmp)+'_fused_straight_reweight_norm'
    #     main(opt)
    #     lamda_tmp += 0.4