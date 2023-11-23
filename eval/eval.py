# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import numpy as np
import data_io
from prf_metrics import cal_prf_ods_metrics, cal_prf_ois_metrics
from segment_metrics import cal_semantic_metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric_mode', type=str, default='prf', help='[prf | sem]')
parser.add_argument('--f1_threshold_mode', type=str, default='ois', help='[ois | ods]')
parser.add_argument('--model_name', type=str, default='crack500_CAM_proportion_c=0.05_lamda=0.7_side1_straight_reweight_norm', help='saved output predicted imgs filename')
parser.add_argument('--results_dir', type=str, default='../result/DeepCrack')
parser.add_argument('--lamda', type=str, default='lamda=0.7')
parser.add_argument('--epoch', type=int, default='10')
parser.add_argument('--proportion_c', type=str, default='c=0.01')
parser.add_argument('--suffix_gt', type=str, default='label_viz', help='Suffix of ground-truth file name')
parser.add_argument('--suffix_pred', type=str, default='fused', help='Suffix of predicted file name')
parser.add_argument('--output', type=str, default='crack500_CAM_proportion_c=0.05_lamda=0.7_side1_straight_reweight_norm_ois.prf')
parser.add_argument('--thresh_step', type=float, default=0.01)
args = parser.parse_args()

if __name__ == '__main__':
    metric_mode = args.metric_mode
    f1_threshold_mode = args.f1_threshold_mode
    model_name = os.path.join(args.model_name, str(args.epoch)+'_Meta_crack500')
    #model_name = args.model_name
    results_dir = os.path.join(args.results_dir, model_name, 'test_latest', 'images')
    src_img_list, tgt_img_list = data_io.get_image_pairs(results_dir, args.suffix_gt, args.suffix_pred)

    final_results = []
    if metric_mode == 'prf':
        if f1_threshold_mode == 'ods':
            final_results = cal_prf_ods_metrics(src_img_list, tgt_img_list, args.thresh_step)
        elif f1_threshold_mode == 'ois':
            final_results = cal_prf_ois_metrics(src_img_list, tgt_img_list, args.thresh_step)
        else:
            print('Error f1_threshold_mode!')
        # else:
        #     # calculate AP
        #     final_results = cal_prf_AP(src_img_list, tgt_img_list, args.thresh_step)

    elif metric_mode == 'sem':
        final_results = cal_semantic_metrics(src_img_list, tgt_img_list, args.thresh_step)
    else:
        print("Unknown mode of metrics.")
    output_path = os.path.join('./prf_results/DeepCrack', model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output = os.path.join(output_path, args.output)
    data_io.save_results(final_results, output)