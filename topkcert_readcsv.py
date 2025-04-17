import numpy as np
import torch
import torch.backends.cudnn as cudnn

import numpy as np
import os
import argparse
import time
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm
import joblib
import seaborn as sns
import statistics


#
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default='checkpoint_drs_new', type=str, help="directory of checkpoints")
parser.add_argument('--data_dir', default='./../../../../public', type=str, help="directory of data")
parser.add_argument('--dataset', default='gtsrb', type=str,
                    choices=('imagenette', 'imagenet', 'cifar', 'cifar100', 'gtsrb', 'flowers102'), help="dataset")
parser.add_argument("--pc_model", default='vit_base_patch16_224_cutout2_128', type=str, help="model name")
parser.add_argument("--drs_model", default='vit_base_patch16_224', type=str, help="model name")
# parser.add_argument("--pc_model", default='resnetv2_50x1_bit_distilled_cutout2_128', type=str, help="model name")
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
parser.add_argument("--override", action='store_true', help='override dumped file')
parser.add_argument("--ablation_size", type=int, default=19, help='override dumped file')
parser.add_argument("--modify", type=bool, default=True, help='override dumped file')
parser.add_argument("--ablation_type", default="column", type=str)
parser.add_argument("--top", default=5, type=int, help="top-k")
parser.add_argument("--vary_patch_size",  type=bool, default=True, help="mode")



args = parser.parse_args()
print(args)
print(args.patch_size)
DATASET = args.dataset
MODEL_DIR = os.path.join('.', args.model_dir)
DATA_DIR = os.path.join(args.data_dir, DATASET)
DUMP_DIR = os.path.join('.', args.dump_dir)
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)
NUM_IMG = args.num_img
PC_MODEL_NAME = args.pc_model
MODEL_NAME = PC_MODEL_NAME
DRS_MODEL_NAME = args.drs_model
ablation_size = args.ablation_size
patch_size = args.patch_size
modify = args.modify
ablation_type=args.ablation_type
TOP_K=args.top
VARY_PATCH_SIZE=args.vary_patch_size
print("top_k: "+str(TOP_K))

total_num_class=10000000000
if DATASET=='imagenet':
    total_num_class=1000
elif DATASET=='cifar':
    total_num_class=10
elif DATASET=='gtsrb':
    total_num_class=43
elif DATASET=='cifar100':
    total_num_class=100
elif DATASET=='flowers102':
    total_num_class = 102
else:
    assert 1==0

num_of_sample=10000000000
if DATASET=='imagenet':
    num_of_sample=50000
elif DATASET=='cifar':
    num_of_sample=10000
elif DATASET=='gtsrb':
    num_of_sample=12630
elif DATASET=='cifar100':
    num_of_sample=10000
elif DATASET=='flowers102':
    num_of_sample = 6149
else:
    assert 1 == 0

#load csv
# np.loadtxt('frame',dtype=np.int,delimiter=None,unpack=False)
if not VARY_PATCH_SIZE:
    if DATASET=='imagenet':
        patch_size=[23,32,39]
    #    112 96 80 64 48
    else:
        patch_size=[14,35]
else:
    # patch_size=[112]
    patch_size=[16,32,48,64,80,96,112]
    # patch_size=[16,32,48]
correct_list=[]
for patch in patch_size:
    correct=np.loadtxt("./6_23_top_k_list_pg_and_topkcert_{}_{}_{}_{}_{}.csv".format(DATASET, DRS_MODEL_NAME,ablation_size,ablation_type, patch),delimiter=",")
    # not_topk=np.loadtxt("./top_k_list_pg_and_topkcert_{}_{}_{}_{}_{}_top_not_1_save.csv".format(DATASET, DRS_MODEL_NAME,ablation_size,ablation_type, patch), delimiter=",")
    correct_list.append(correct)

for correct in correct_list:
    pg_correct=0
    pg_cert=0

    tk_correct=0
    tk_cert=0

    pg_output_rank=correct[:,0]
    pg_robust_rank=correct[:,1]
    tk_output_rank=correct[:,2]
    tk_robust_rank=correct[:,3]
    pgt_output_rank=correct[:,4]
    pgt_robust_rank=correct[:,5]

    pg_correct=np.sum(pg_output_rank<=TOP_K)
    round(pg_correct, 2)
    pg_cert=np.sum(pg_robust_rank<=TOP_K)
    round(pg_cert, 2)

    tk_correct=np.sum(tk_output_rank<=TOP_K)
    round(tk_correct, 2)
    tk_cert=np.sum(tk_robust_rank<=TOP_K)
    round(tk_cert, 2)

    pgt_correct=np.sum(pgt_output_rank<=TOP_K)
    round(pgt_correct, 2)
    pgt_cert=np.sum(pgt_robust_rank<=TOP_K)
    round(pgt_cert, 2)


    print(tk_robust_rank.mean())
    print(pg_robust_rank.mean())
    print(pgt_robust_rank.mean())



    print("pgt_correct "+str(pgt_correct*100/num_of_sample))
    print("pgt_cert "+str(pgt_cert*100/num_of_sample))
    print("pg_correct "+str(pg_correct*100/num_of_sample))
    print("pg_cert "+str(pg_cert*100/num_of_sample))
    print("tk_correct "+str(tk_correct*100/num_of_sample))
    print("tk_cert "+str(tk_cert*100/num_of_sample))
    print("\n")


def line_plot(correct_list):
    counter=0
    plt.figure(figsize=(8, 4.8))
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]

        clean_full=np.zeros(shape=(total_num_class+1, 1))
        # print(pg_output_rank)
        top_k, cnt = np.unique(pg_output_rank, return_counts=True)
        for idx in range(total_num_class):
            idx=idx+1
            num = 0
            if idx in top_k:
                position=np.where(top_k==idx)
                num=cnt[position]
            clean_full[idx]=clean_full[idx-1]+num

        tc_full=np.zeros(shape=(total_num_class+1, 1))
        # print(pg_robust_rank)
        top_k, cnt = np.unique(pg_robust_rank, return_counts=True)
        for idx in range(total_num_class):
            idx=idx+1
            num=0
            if idx in top_k:
                position=np.where(top_k==idx)
                num=cnt[position]
            tc_full[idx]=tc_full[idx-1]+num

        pg_full=np.zeros(shape=(total_num_class+1, 1))
        # print(tk_robust_rank)
        top_k, cnt = np.unique(tk_robust_rank, return_counts=True)
        for idx in range(total_num_class):
            idx=idx+1
            num=0
            if idx in top_k:
                position=np.where(top_k==idx)
                num=cnt[position]
            pg_full[idx]=pg_full[idx-1]+num

        pgt_full=np.zeros(shape=(total_num_class+1, 1))
        # print(tk_robust_rank)
        top_k, cnt = np.unique(pgt_robust_rank, return_counts=True)
        for idx in range(total_num_class):
            idx=idx+1
            num=0
            if idx in top_k:
                position=np.where(top_k==idx)
                num=cnt[position]
            pgt_full[idx]=pgt_full[idx-1]+num

        if DATASET=='imagenet':
            if counter == 0:
                plt.plot(range(1, total_num_class + 1), clean_full[1:total_num_class + 1] / num_of_sample, ':',
                         linewidth=2, markersize=10, color='black', label='Clean Accuracy')
                plt.plot(range(1, total_num_class + 1), pg_full[1:total_num_class + 1] / num_of_sample, '-',
                         linewidth=2, markersize=18, color='green', markerfacecolor='white',
                         label='Certified Accuracy of TC against 1% patch')
                plt.plot(range(1, total_num_class + 1), tc_full[1:total_num_class + 1] / num_of_sample, '--',
                         linewidth=2, markersize=10, color='green',
                         label='Certified Accuracy of PG$_{\spadesuit}$ against 1% patch ')
                plt.plot(range(1, total_num_class + 1), pgt_full[1:total_num_class + 1] / num_of_sample, '.',
                         linewidth=2, markersize=10, color='green',
                         label='Certified Accuracy of PG against 1 patch ')

            if counter == 1:
                plt.plot(range(1, total_num_class + 1), clean_full[1:total_num_class + 1] / num_of_sample, ':',
                         linewidth=2, markersize=10, color='black')
                plt.plot(range(1, total_num_class + 1), pg_full[1:total_num_class + 1] / num_of_sample, '-',
                         linewidth=2, markersize=18, color='orange', markerfacecolor='white',
                         label='Certified Accuracy of TC against 2% patch')
                plt.plot(range(1, total_num_class + 1), tc_full[1:total_num_class + 1] / num_of_sample, '--',
                         linewidth=2, markersize=10, color='orange',
                         label='Certified Accuracy of PG$_{\spadesuit}$ against 2% patch')
                plt.plot(range(1, total_num_class + 1), pgt_full[1:total_num_class + 1] / num_of_sample, '.',
                         linewidth=2, markersize=10, color='orange',
                         label='Certified Accuracy of PG against 2% patch ')
            if counter==2:
                plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black')
                plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='blue',label='Certified Accuracy of TC against 3% patch')
                plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10, color='blue',label='Certified Accuracy of PG$_{\spadesuit}$ against 3% patch')
                plt.plot(range(1,total_num_class+1), pgt_full[1:total_num_class+1]/num_of_sample, '.', linewidth=2, markersize=10, color='blue',label='Certified Accuracy of PG_t$_{\spadesuit}$ against 3% patch ')

        else:
            if counter==0:
                plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black',label='Clean Accuracy')
                plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='green',markerfacecolor='white',label='Certified Accuracy of TC against 0.4% patch')
                plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10, color='green',label='Certified Accuracy of PG$_{\spadesuit}$ against 0.4% patch ')
                plt.plot(range(1,total_num_class+1), pgt_full[1:total_num_class+1]/num_of_sample, '.', linewidth=2, markersize=10, color='green',label='Certified Accuracy of PG against 0.4% patch ')

            if counter==1:
                plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black')
                plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='orange',markerfacecolor='white',label='Certified Accuracy of TC against 2.4% patch')
                plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10,color='orange',label='Certified Accuracy of PG$_{\spadesuit}$ against 2.4% patch')
                plt.plot(range(1,total_num_class+1), pgt_full[1:total_num_class+1]/num_of_sample, '.', linewidth=2, markersize=10, color='orange',label='Certified Accuracy of PG against 2.4% patch ')


        # if counter==0:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black',label='Clean Accuracy')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='green',markerfacecolor='white',label='Certified Accuracy of TC against 1% patch')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10, color='green',label='Certified Accuracy of PG$_{\spadesuit}$ against 1% patch ')
        #     plt.plot(range(1, total_num_class + 1), pgt_full[1:total_num_class + 1] / num_of_sample, '.', linewidth=2,
        #          markersize=10, color='green', label='Certified Accuracy of PG_t$_{\spadesuit}$ against 0.4% patch ')
        # if counter==1:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='orange',markerfacecolor='white',label='Certified Accuracy of TC against 2% patch')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10,color='orange',label='Certified Accuracy of PG$_{\spadesuit}$ against 2% patch')
        #     plt.plot(range(1, total_num_class + 1), pgt_full[1:total_num_class + 1] / num_of_sample, '.', linewidth=2,
        #          markersize=10, color='orange', label='Certified Accuracy of PG_t$_{\spadesuit}$ against 0.4% patch ')
        #
        # if counter==2:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, ':', linewidth=2, markersize=10, color='black')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '-', linewidth=2, markersize=18, color='blue',label='Certified Accuracy of TC against 3% patch')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '--', linewidth=2, markersize=10, color='blue',label='Certified Accuracy of PG$_{\spadesuit}$ against 3% patch')
        #     plt.plot(range(1,total_num_class+1), pgt_full[1:total_num_class+1]/num_of_sample, '.', linewidth=2, markersize=10, color='blue',label='Certified Accuracy of PG_t$_{\spadesuit}$ against 0.4% patch ')

        plt.xscale('log')
        # plt.xlabel('Topk',labelpad=-1.5)
        # plt.xlabel('Value of k in Topk',labelpad=-8,size=11)
        plt.xlabel('Value of k in Topk',labelpad=0,size=11)

        plt.ylabel('Accuracy',size=12)
        if DATASET=='cifar':
            dataset_name='CIFAR10'
        if DATASET=='cifar100':
            dataset_name='CIFAR100'
        if DATASET=='gtsrb':
            dataset_name='GTSRB'
        if DATASET=='imagenet':
            dataset_name='ImageNet'
        # plt.title('Topk Accuracy against different patch sizes across different k for {}'.format(dataset_name))
        plt.legend(frameon=False,bbox_to_anchor=(0.5, -0.4),loc=8,ncol=2, prop = {'size':10.2})
        # plt.gcf().subplots_adjust(bottom=0.25)
        # plt.title('Topk Accuracy against different patch sizes across different k for {}'.format(dataset_name))
        # plt.legend(frameon=False,bbox_to_anchor=(0.5, -0.4),loc=8,ncol=2, prop = {'size':9.7})
        plt.gcf().subplots_adjust(bottom=0.25)
        # if counter==0:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, '.:', linewidth=2, markersize=10, color='blue')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '*-', linewidth=3, markersize=18, color='green',markerfacecolor='white')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '^-', linewidth=3, markersize=10, color='orange')
        # if counter==1:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, '.:', linewidth=2, markersize=10, color='blue')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '*-', linewidth=3, markersize=18, color='green',markerfacecolor='white')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '^-', linewidth=3, markersize=10,color='orange')
        # if counter==2:
        #     plt.plot(range(1,total_num_class+1), clean_full[1:total_num_class+1]/num_of_sample, '.-.', linewidth=2, markersize=10, color='blue')
        #     plt.plot(range(1,total_num_class+1), pg_full[1:total_num_class+1]/num_of_sample, '*--', linewidth=3, markersize=18, color='green')
        #     plt.plot(range(1,total_num_class+1), tc_full[1:total_num_class+1]/num_of_sample, '^--', linewidth=3, markersize=10, color='orange')
        # plt.xscale('log')
        # plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
        counter=counter+1

    # plt.grid(which='both')
    plt.show()



def box_plot(correct_list):
    # fig, ax = plt.subplots(1, 4, figsize=(8, 4.8))
    # ax[0].set_title('Certified samples by PG$_{\spadesuit}$', size=12)
    # ax[0].set_ylabel('Value of k in Topk', size=12)
    # ax[1].set_title('Certified samples by TC', size=12)
    # ax[2].set_title('Correct Samples', size=12)
    # ax[3].set_title('{PG}', size=12)
    fig, ax = plt.subplots(1, 3, figsize=(8, 4.4))
    ax[0].set_title('Certified samples by PG$_{\spadesuit}$-ViT', size=10.5)
    ax[0].set_ylabel('Value of k in Topk', size=12)
    ax[2].set_title('Correct Samples', size=10.5)
    ax[1].set_title('Certified samples by PG-ViT', size=10.5)

    if DATASET=='imagenet':
        yticks =ax[0].get_yticklabels()
        for ytick in yticks:
            ytick.set_fontsize(7)
        yticks =ax[1].get_yticklabels()
        for ytick in yticks:
            ytick.set_fontsize(7)
        yticks =ax[2].get_yticklabels()
        for ytick in yticks:
            ytick.set_fontsize(7)

    for correct in correct_list:
        # 创建箱型图
        violin_parts_0 =ax[0].violinplot((correct[:,1]),showmeans=True)
        # violin_parts_0.set_title('1')

        # for pc in violin_parts_0['bodies']:
        #     pc.set_facecolor('blue')
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(1)
        # ax[0].set_yscale('log')

        # violin_parts_1 =ax[1].violinplot((correct[:,3]),showmeans=True)
        # violin_parts_2 =ax[2].violinplot((correct[:,0]),showmeans=True)
        # violin_parts_3 =ax[3].violinplot((correct[:,5]),showmeans=True)

        violin_parts_3 =ax[2].violinplot((correct[:,0]),showmeans=True)
        violin_parts_2 =ax[1].violinplot((correct[:,5]),showmeans=True)


        # ax[1].set_yscale('log')
    # plt.gcf().subplots_adjust(bottom=0.3)
    # plt.legend()
    if DATASET=='imagenet':
        green_patch = mpatches.Patch(color='green', label='Patch with size 3%')
        orange_patch = mpatches.Patch(color='orange', label='Patch with size 2%')
        blue_patch = mpatches.Patch(color='blue', label='Patch with size 1%')
        fig.legend(frameon=False,bbox_to_anchor=(0.5, -0.02),loc=8,ncol=3, prop = {'size':12},handles=[blue_patch,orange_patch,green_patch ])

    else:
        orange_patch = mpatches.Patch(color='orange', label='Patch size 2.4%')
        blue_patch = mpatches.Patch(color='blue', label='Patch size 0.4%')
        # fig.legend(frameon=False,bbox_to_anchor=(0.5, -0.02),loc=8,ncol=3, prop = {'size':12},handles=[blue_patch,orange_patch,green_patch ])
        fig.legend(frameon=False,bbox_to_anchor=(0.5, -0.02),loc=8,ncol=3, prop = {'size':12},handles=[blue_patch,orange_patch ])
    if DATASET == 'cifar':
        dataset_name = 'CIFAR10'
    if DATASET == 'cifar100':
        dataset_name = 'CIFAR100'
    if DATASET == 'gtsrb':
        dataset_name = 'GTSRB'
    if DATASET == 'imagenet':
        dataset_name = 'ImageNet'
    # plt.suptitle('Distribution of samples in {}'.format(dataset_name), size=25)
    # plt.yticks(fontsize=1)
    plt.show()
    # fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    # for idx in range(len(correct_list)):
    #     创建箱型图
        # ax[idx].violinplot((correct_list[idx][:,1]), showmeans=True)
        # ax[idx].set_yscale('log')
        # ax[idx].violinplot((correct_list[idx][:,3]), showmeans=True)
        # ax[idx].set_yscale('log')
    # plt.show()

def motivate_line_1 (correct_list):
    counter=0
    fig=plt.figure(figsize=(3.8, 3.5))
    ax = fig.add_subplot(111)
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # plt.tight_layout()
    plt.subplots_adjust(left=0.15,right=0.9)
    pgt_robust_rank_list=[]
    pgt_robust_rank_list=[]
    baseline=0
    x_list=[32,48,64,80,96,112]
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]
        if counter==0:
            baseline=pgt_robust_rank
            counter = counter + 1

            continue
        pgt_robust_rank_list.append(pgt_robust_rank/baseline)
        baseline = pgt_robust_rank
        counter=counter+1

    mean=[]
    for pgt_robust_rank in pgt_robust_rank_list:
        mean.append(statistics.mean(pgt_robust_rank))
    plt.plot(x_list,mean, 'go-', label='PG')
    print("pgt_robust_rank"+str(mean))

    pgt_robust_rank_list=[]
    baseline=0
    counter=0
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]
        if counter==0:
            baseline=pg_robust_rank
            counter = counter + 1
            continue
        pgt_robust_rank_list.append(pg_robust_rank/baseline)
        baseline = pg_robust_rank
        counter=counter+1
    mean=[]
    for pgt_robust_rank in pgt_robust_rank_list:
        mean.append(statistics.mean(pgt_robust_rank))
    # fig.yaxis.tick_right()
    plt.plot(x_list,mean, 'bo-', label='PG$_{\spadesuit}$')
    print("pg_robust_rank"+str(mean))

    # pgt_robust_rank_list=[]
    # baseline=0
    # counter=0
    # for correct in correct_list:
    #     pg_output_rank = correct[:, 0]
    #     pg_robust_rank = correct[:, 1]
    #     tk_output_rank = correct[:, 2]
    #     tk_robust_rank = correct[:, 3]
    #     pgt_output_rank = correct[:, 4]
    #     pgt_robust_rank = correct[:, 5]
    #     if counter==0:
    #         baseline=tk_robust_rank
    #         counter = counter + 1
    #         continue
    #     pgt_robust_rank_list.append(tk_robust_rank/baseline)
    #     baseline = tk_robust_rank
    #     counter=counter+1
    # mean=[]
    # for pgt_robust_rank in pgt_robust_rank_list:
    #     mean.append(statistics.mean(pgt_robust_rank))
    # plt.plot(x_list,mean, 'ro-', label='CC')
    # print("pg_robust_rank"+str(mean))
    # plt.yscale("log")

    plt.xlabel('Patch Size',labelpad=0,size=13)
    plt.ylabel('Sensitivity',labelpad=0,size=13)
    plt.legend(prop={'size': 12})
    plt.xticks(x_list)
    # motivate_line_2(correct_list)
    plt.show()

def motivate_line_2 (correct_list):
    counter=0
    # plt.figure(figsize=(4.1, 3.2))
    # f, (top, bottom) =plt.subplots(figsize=(8, 3.5),nrows=2)
    f, (top, bottom) =plt.subplots(figsize=(3.8, 3.5),nrows=2)
    plt.subplots_adjust(left=0.18,right=0.95)
    # plt.subplots_adjust(wspace=0, hspace=0.35)
    # plt.subplots_adjust(wspace=0, hspace=0.01)

    pgt_robust_rank_list=[]
    baseline=0
    x_list=[16,32,48,64,80,96,112]
    x_list_str = [str(32), str(48), str(64), str(80), str(96), str(112)]
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]
        if counter==0:
            baseline=1
            # counter = counter + 1
            #
            # continue
        pgt_robust_rank_list.append(pgt_robust_rank/baseline)
        # baseline = pgt_output_rank
        counter=counter+1

    mean=[]
    for pgt_robust_rank in pgt_robust_rank_list:
        mean.append(statistics.mean(pgt_robust_rank))
    plt.subplot(2, 1, 1)
    plt.plot(x_list,mean, 'g^-', markersize=9)
    print("pgt_robust_rank"+str(mean))
    plt.boxplot(pgt_robust_rank_list,positions=x_list,widths=10,
                # medianprops={'color': 'red', 'linewidth': '1.5'},
                # patch_artist=True,
                meanline=False,
                showmeans=False,
                medianprops={'color': 'orange', 'ls': '-', 'linewidth': '3'},
                # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                showfliers=True,
                # flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                # notch=True,
                )
    plt.ylabel('Min k',labelpad=0,size=15)
    plt.title('PG',size=15,x=0.88,y=0)


    pgt_robust_rank_list=[]
    baseline=0
    counter=0
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]
        if counter==0:
            baseline=1
            # counter = counter + 1
            # continue
        pgt_robust_rank_list.append(pg_robust_rank/baseline)
        # baseline = pg_output_rank
        counter=counter+1
    plt.subplot(2, 1, 2)
    # bottom.relim()
    plt.boxplot(pgt_robust_rank_list,positions=x_list,widths=10,
                # medianprops={'color': 'red', 'linewidth': '1.5'},
                # patch_artist=True,
                meanline=False,
                showmeans=False,
                medianprops={'color': 'orange', 'ls': '-', 'linewidth': '3'},

                # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
                showfliers=True,
                # flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
                # notch=True,
                )
    mean=[]
    for pgt_robust_rank in pgt_robust_rank_list:
        mean.append(statistics.mean(pgt_robust_rank))
    plt.plot(x_list,mean, 'b^-', markersize=9)
    print("pg_robust_rank"+str(mean))
    # plt.yscale("log")
    plt.xlabel('Patch Size',labelpad=0,size=13)
    plt.ylabel('Min k',labelpad=0,size=15)
    plt.title('PG$_{\spadesuit}$',size=15,x=0.88,y=0)
    # plt.legend()
    plt.show()

def patch_size_line_plot(correct_list):
    counter = 0
    plt.figure(figsize=(6, 3))
    plt.subplots_adjust(bottom=0.16)
    x_list = [16, 32, 48, 64, 80, 96, 112]
    pg_output_rank_list = []
    pg_robust_rank_list = []
    tk_output_rank_list = []
    tk_robust_rank_list = []
    pgt_output_rank_list = []
    pgt_robust_rank_list = []
    for correct in correct_list:
        pg_output_rank = correct[:, 0]
        pg_robust_rank = correct[:, 1]
        tk_output_rank = correct[:, 2]
        tk_robust_rank = correct[:, 3]
        pgt_output_rank = correct[:, 4]
        pgt_robust_rank = correct[:, 5]

        pg_output_rank=pg_output_rank[pg_output_rank<=TOP_K]
        pg_robust_rank=pg_robust_rank[pg_robust_rank<=TOP_K]
        tk_output_rank=tk_output_rank[tk_output_rank<=TOP_K]
        tk_robust_rank=tk_robust_rank[tk_robust_rank<=TOP_K]
        pgt_output_rank=pgt_output_rank[pgt_output_rank<=TOP_K]
        pgt_robust_rank=pgt_robust_rank[pgt_robust_rank<=TOP_K]

        pg_output_rank_list.append(len(pg_output_rank)/num_of_sample)
        pg_robust_rank_list.append(len(pg_robust_rank)/num_of_sample)
        tk_output_rank_list.append(len(tk_output_rank)/num_of_sample)
        tk_robust_rank_list.append(len(tk_robust_rank)/num_of_sample)
        pgt_output_rank_list.append(len(pgt_output_rank)/num_of_sample)
        pgt_robust_rank_list.append(len(pgt_robust_rank)/num_of_sample)

    plt.plot(x_list, pg_output_rank_list[0:len(pg_output_rank_list)],  marker='o',linestyle="solid",markerfacecolor='white',color='green',
             linewidth=2, markersize=10, label='PG$_\spadesuit-acc_{clean}$')
    plt.plot(x_list, pg_robust_rank_list[0:len(pg_robust_rank_list)], marker='o',linestyle="dotted",color='green',
             linewidth=2, markersize=10, label='PG$_\spadesuit-acc_{cert}$')
    plt.plot(x_list, tk_output_rank_list[0:len(tk_output_rank_list)], marker='*',linestyle="solid",markerfacecolor='white',color='red',
             linewidth=2, markersize=15, label='CC$-acc_{clean}$')
    plt.plot(x_list, tk_robust_rank_list[0:len(tk_robust_rank_list)], marker='*',linestyle="dotted",color='red',
             linewidth=2, markersize=15, label='CC$-acc_{cert}$')
    plt.plot(x_list, pgt_output_rank_list[0:len(pgt_output_rank_list)], marker='^',linestyle="solid",markerfacecolor='white',color='blue',
             linewidth=2, markersize=10, label='PG$-acc_{clean}$')
    plt.plot(x_list, pgt_robust_rank_list[0:len(pgt_robust_rank_list)], marker='^',linestyle="dotted",color='blue',
             linewidth=2, markersize=10, label='PG$-acc_{cert}$')
    plt.legend(prop={'size': 7})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel("acc (k=10)",fontsize=15)
    plt.xlabel("Patch Size",fontsize=13)
    plt.title("GTSRB")
    plt.show()
    print(tk_robust_rank_list)
patch_size_line_plot(correct_list)
# box_plot(correct_list)
# motivate_line_2 (correct_list)
# # 绘制直方图
# plt.hist(pg_array, bins=10, label='Data 1')
# plt.hist(tc_array, bins=10, label='Data 2')
#
# # 设置图表属性
# plt.title('RUNOOB hist() TEST')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
#
# # 显示图表
# plt.show()
