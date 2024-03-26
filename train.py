import os
from net import Mnet
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from lib.dataset_lable import Data
from lib.data_prefetcher_label import DataPrefetcher
from torch.nn import functional as F

from discriminator_ce import FCDiscriminator
from cm import Curriculum

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def loss_1(score1, score2, score3, score4, label):

    score1 = F.interpolate(score1, label.shape[2:], mode='bilinear', align_corners=True)
    score2 = F.interpolate(score2, label.shape[2:], mode='bilinear', align_corners=True)
    score3 = F.interpolate(score3, label.shape[2:], mode='bilinear', align_corners=True)
    score4 = F.interpolate(score4, label.shape[2:], mode='bilinear', align_corners=True)


    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')

    return sal_loss1 + sal_loss2 + sal_loss3 + sal_loss4


if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    # dataset

    type1_root = '/dataset/Type_I/'
    type2_root = '/dataset/Type_II/'
    steel_root = 'dataset/SD-saliency-900/'
    neu1_root = '/dataset/neu_1/'
    neu2_root = '/dataset/neu_2/'

    save_path = './model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    lr_d = 0.001
    lr_cm = 0.0001
    lambda_adv_target = 0.01
    batch_size = 8
    epoch = 130
    lr_dec = [130, 145]

    #   TYPE1:0 ; TYPE2:1;  NEU113: 2; NET2 :3 ;STEEL:4; TILE:5

    type1_data = Data(type1_root, 0)
    type2_data = Data(type2_root, 1)
    neu1_data = Data(neu1_root, 2)
    neu2_data = Data(neu2_root, 3)
    steel_data = Data(steel_root, 4)

    #  init net
    net = Mnet().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    net.train()

    #  init dis
    model_D = FCDiscriminator()
    model_D.train()
    model_D.cuda()
    optimizer_D = optim.Adam(model_D.parameters(), lr=lr_d, betas=(0.9, 0.99))
    bce_loss = torch.nn.CrossEntropyLoss()

    #  init cm
    weight_cm = Curriculum()
    optimizer_cm = optim.Adam(weight_cm.parameters(), lr=lr_cm, betas=(0.9, 0.99))
    weight_cm.cuda()
    weight_cm.train()

    num_params = 0
    for p in net.parameters():
        num_params += p.numel()
    print(num_params)

    for epochi in range(1, epoch + 1):
        if epochi in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,
                                  momentum=0.9)

            print(lr)

        domain_dataset_list = [type2_data, neu1_data, neu2_data, steel_data]

        domain_list = np.random.permutation(4)
        meta_train_domain_list = domain_list[:3]
        meta_test_domain_list = domain_list[3]
        meta_train_dataset = ConcatDataset([domain_dataset_list[meta_train_domain_list[0]],
                                            domain_dataset_list[meta_train_domain_list[1]],
                                            domain_dataset_list[meta_train_domain_list[2]],
                                            ])
        meta_train_loader = DataLoader(meta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        prefetcher = DataPrefetcher(meta_train_loader)
        rgb, label, cla = prefetcher.next()

        meta_test_dataset = domain_dataset_list[meta_test_domain_list]
        train_len = len(meta_train_dataset)
        test_len = len(meta_test_dataset)
        new_test_data = meta_test_dataset
        for k in range(train_len // test_len + 1):
            new_test_data = ConcatDataset([new_test_data, meta_test_dataset])
        meta_test_dataset_final = new_test_data
        meta_test_loader = DataLoader(meta_test_dataset_final, batch_size=batch_size, shuffle=True, num_workers=4)
        test_prefetcher = DataPrefetcher(meta_train_loader)
        test_rgb, test_label, test_cla = test_prefetcher.next()

        print(train_len)
        print(test_len)
        print(len(meta_train_loader))
        print(len(meta_test_loader))
        iter_num = len(meta_train_loader)


        train_sal_loss = 0
        loss_adv_value = 0
        loss_D_value = 0
        test_sal_loss = 0
        net.zero_grad()
        i = 0

        for j in range(iter_num):
            i += 1

            for param in model_D.parameters():
                param.requires_grad = False

            score1, score2, score3, score4, s1, s2, s3, s4 = net(rgb)
            train_loss = loss_1(score1, score2, score3, score4, s1, s2, s3, s4, label)
            score_d = F.interpolate(score4, size=[352, 352], mode='bilinear', align_corners=True)
            train_sal_loss += train_loss.data
            train_loss.backward()

            w = weight_cm(rgb)

            D_out = model_D(score_d)
            loss_adv_train = bce_loss(D_out, cla.long())

            loss_adv = lambda_adv_target * loss_adv_train
            loss_adv.backward()
            loss_adv_value += loss_adv.data

            for param in model_D.parameters():
                param.requires_grad = True

            score_d = score_d.detach()
            D_out = model_D(score_d)

            loss_D_source = bce_loss(D_out, cla.long())
            loss_D = w * loss_D_source
            loss_D.backward(loss_D.clone().detach())
            loss_D_value += loss_D_source.data

            t_score1, t_score2, t_score3, t_score4, t_s1, t_s2, t_s3, t_s4 = net(test_rgb)
            test_loss = loss_1(t_score1, t_score2, t_score3, t_score4, t_s1, t_s2, t_s3, t_s4, test_label)
            test_sal_loss += test_loss.data
            test_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            optimizer_D.step()
            model_D.zero_grad()
            optimizer_cm.step()
            weight_cm.zero_grad()

            rgb, label, cla = prefetcher.next()
            test_rgb, test_label, test_cla = test_prefetcher.next()

        if epochi >= 70 and epochi % 10 == 0:
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/final.pth' % (save_path))
