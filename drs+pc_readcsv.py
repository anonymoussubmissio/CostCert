import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time
from tqdm import tqdm
import joblib
import time

from utils.topkcert_util import certified_drs_four_delta, certified_drs_three_position, \
    certified_drs_three_position_mixup, certified_drs_new_version, \
    certified_drs_pg_version_ablated
from utils.new import majority_of_mask_single, certified_nowarning_detection, \
    certified_warning_detection, warning_detection, certified_warning_drs, majority_of_drs_single, certified_drs, \
    pc_malicious_label, warning_drs, double_masking_precomputed_with_case_num, warning_analysis, malicious_list_drs, \
    malicious_list_compare, pc_malicious_label_with_location, mask_ablation_for_all, \
    suspect_column_list_cal, certified_with_location, check_maskfree_empty, suspect_column_list_cal_fix, \
    pc_malicious_label_check, double_masking_precomputed_with_case_num_modify, warning_analysis_modify, certified_pg
from utils.pd import one_masking_statistic, double_masking_detection, double_masking_detection_nolemma1
from utils.setup import get_model, get_data_loader
from utils.defense import gen_mask_set, double_masking_precomputed, certify_precomputed

#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoint_drs_new', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='./../../../../public', type=str, help="directory of data")
parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'svhn', 'flower102',"gtsrb"), help="dataset")
parser.add_argument("--pc_model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
parser.add_argument("--drs_model", default='vit_base_patch16_224', type=str, help="model name")
# parser.add_argument("--pc_model", default='resnetv2_50x1_bit_distilled', type=str, help="model name")
# parser.add_argument("--drs_model", default='resnetv2_50x1_bit_distilled', type=str, help="model name")

parser.add_argument("--num_img", default=-1, type=int,
                    help="number of randomly selected images for this experiment (-1: using the all images)")
parser.add_argument("--mask_stride", default=-1, type=int, help="mask stride s (square patch; conflict with num_mask)")
parser.add_argument("--num_mask", default=6, type=int,
                    help="number of mask in one dimension (square patch; conflict with mask_stride)")
parser.add_argument("--patch_size", default=32, type=int, help="size of the adversarial patch (square patch)")
parser.add_argument("--pa", default=-1, type=int,
                    help="size of the adversarial patch (first axis; for rectangle patch)")
parser.add_argument("--pb", default=-1, type=int,
                    help="size of the adversarial patch (second axis; for rectangle patch)")
parser.add_argument("--dump_dir", default='dump', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--dump_dir_pc", default='/home/qlzhou4/pd/dump_majorrevision', type=str, help='directory to dump two-mask predictions')
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size", type=int, default=19, help='override dumped file')
parser.add_argument("--modify", type=bool, default=True, help='override dumped file')
parser.add_argument("--ablation_type", default="column", type=str)
parser.add_argument("--top", default=5, type=int, help="top-k")



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_args()
print(args)
print(args.patch_size)
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
DUMP_PC_DIR = args.dump_dir_pc
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
NUM_IMG = args.num_img
# PC_MODEL_NAME
PC_MODEL_NAME = args.pc_model
# MODEL_NAME = PC_MODEL_NAME
MODEL_NAME = args.drs_model
DRS_MODEL_NAME = args.drs_model
ablation_size = args.ablation_size
patch_size = args.patch_size
modify = args.modify
ablation_type=args.ablation_type
TOP_K=args.top

model = get_model(MODEL_NAME, DATASET, MODEL_DIR,ablation_size=ablation_size, ablation_type=ablation_type)
# # print(model.named_parameters())
val_loader, NUM_IMG, ds_config = get_data_loader(DATASET, DATA_DIR, model, batch_size=16, num_img=NUM_IMG, train=False)
#
# device = 'cuda'
# # model = model.to(device)
# model.eval()
cudnn.benchmark = True

# generate the mask set

# generate the mask set
mask_list, MASK_SIZE, MASK_STRIDE = gen_mask_set(args, ds_config)

label_list = joblib.load(
    os.path.join(DUMP_DIR, "label_list_{}_{}_{}_drs_{}_{}_{}_{}_drs.z".format(DATASET, DRS_MODEL_NAME,DATASET,ablation_size,DATASET,ablation_type,NUM_IMG)))
# label_list_cifar_vit_base_patch16_224_cifar_drs_19_cifar_column_10000_drs.z

# label_list_cifar_vit_base_patch16_224_cutout2_128_10000.z'
prediction_map_list_drs = joblib.load(os.path.join(DUMP_DIR,
                                                   "prediction_map_list_drs_two_mask_{}_{}_{}_drs_{}_{}_{}_m{}_s1_{}.z".format(
                                                       DATASET, DRS_MODEL_NAME, DATASET,ablation_size,DATASET,ablation_type,ablation_size,
                                                       NUM_IMG)))

prediction_map_list_pc = joblib.load(os.path.join(DUMP_PC_DIR,
                                                  "prediction_map_list_two_mask_{}_{}_m{}_s{}_{}.z".format(DATASET,
                                                                                                           PC_MODEL_NAME,
                                                                                                           str(MASK_SIZE),
                                                                                                           str(MASK_STRIDE),
                                                                                                           NUM_IMG)))

# prediction_map_list_drs_two_mask_imagenet_vit_base_patch16_224_imagenet_drs_37_m37_imagenet_column_s1_50000.z'
#
# prediction_map_list_drs_two_mask_imagenet_vit_base_patch16_224_imagenet_drs_37_imagenet_column_m37_s1_50000.z
# prediction_map_list_drs_two_mask_imagenet_vit_base_patch16_224_imagenet_drs_37_m37_s1_50000.z


def static_cert_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    # if output_label_pc == output_label_drs and (robust_pc or robust_drs):
    #     return True
    # # elif robust_pc:
    # #     return True
    # if output_label_pc == output_label_drs and (robust_pc or robust_drs):
    #     return True
    # elif robust_pc:
    #     return True
    if robust_pc or (output_label_pc == output_label_drs and robust_drs):
        return True
    return False


def static_cert_very_stable_analysis(output_label_pc, output_label_drs, robust_pc, robust_drs):
    if output_label_pc == output_label_drs and (robust_pc and robust_drs):
        return True
    return False

total_num_class=1000000
if DATASET=='imagenet':
    total_num_class=1000
elif DATASET=='cifar':
    total_num_class=10
elif DATASET=='cifar100':
    total_num_class=100
elif DATASET=='gtsrb':
    total_num_class=43
elif DATASET=='flowers102':
    total_num_class=102
else:
    print("warning!!!")


pg_correct=0
pg_cert=0
new_correct=0
new_cert=0
clean=0
clean_pc=0

pg_cert_top1=0
drs_cert_top1=0
all_correct = np.zeros(shape=(NUM_IMG, 4))
T_PG=0
T_TC=0

robust_count=0
k=5
patch=args.patch_size
correct = np.loadtxt(
    "./6_23_top_k_list_pg_and_topkcert_{}_{}_{}_{}_{}.csv".format(DATASET, DRS_MODEL_NAME, ablation_size, ablation_type,
                                                                  patch), delimiter=",")

# top_not_1_save = np.zeros(shape=(NUM_IMG, 2))
# np.savetxt("./top_k_list_pg_and_topkcert_{}_{}_{}_{}.csv".format(DATASET, DRS_MODEL_NAME,ablation_size,ablation_type), top_k_save,fmt="%.2f",delimiter=",")
for i, (label, prediction_map_drs,prediction_map_pc) in enumerate(
        zip(label_list, prediction_map_list_drs,prediction_map_list_pc)):
    # calculate the majority
    # output_label_drs_column_four_delta, robust_drs_column_three_position = certified_drs_three_position(
    #     prediction_map_drs_column, ablation_size,
    #     patch_size)
    # output_label_drs_row_four_delta, robust_drs_row_three_position = certified_drs_three_position(
    #     prediction_map_drs_row, ablation_size, patch_size)
    # output_label_drs_column, robust_drs_column = certified_drs(prediction_map_drs_column, ablation_size,
    #                                                            patch_size)
    # output_label_drs_row, robust_drs_row = certified_drs(prediction_map_drs_row, ablation_size, patch_size)
    prediction_map_pc = prediction_map_pc + prediction_map_pc.T - np.diag(np.diag(prediction_map_pc))
    output_label_pc, case_num = double_masking_precomputed_with_case_num(prediction_map_pc)
    robust_pc = certify_precomputed(prediction_map_pc, output_label_pc)
    tc_result_output = correct[i, 2]
    tc_result_cert = correct[i, 3]
    if output_label_pc == label:
        clean_pc = clean_pc + 1
        clean = clean +1
    elif tc_result_output<=TOP_K:
        clean = clean + 1
    if robust_pc:
        robust_count=robust_count+1
    if not robust_pc:
        if tc_result_cert<=TOP_K:
            new_cert=new_cert+1





    print("clean " + str(clean) + ' ' + str(clean / NUM_IMG))
    print("clean_pc " + str(clean_pc) + ' ' + str(clean_pc / NUM_IMG))


    # print("pg_correct " + str(pg_correct) + ' ' + str(pg_correct / NUM_IMG))
    # print("pg_cert " + str(pg_cert) + ' ' + str(pg_cert / NUM_IMG))
    # print("new_correct " + str(new_correct) + ' ' + str(new_correct / NUM_IMG))
    print("new_cert " + str(new_cert) + ' ' + str(new_cert / NUM_IMG))
    # print("drs_cert_top1 " + str(drs_cert_top1) + ' ' + str(drs_cert_top1 / NUM_IMG))
    # print("pg_cert_top1 " + str(pg_cert_top1) + ' ' + str(pg_cert_top1 / NUM_IMG))
    print("robust_count" + str(robust_count) + ' ' + str(robust_count / NUM_IMG))
    print("robust_count+new" + str(robust_count+new_cert) + ' ' + str((robust_count+new_cert) / NUM_IMG))
    print("\n")