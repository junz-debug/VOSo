#import seaborn as sns
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
import os 
#from tqdm import tqdm
from torch.nn import functional as F
from get_cam_mask import get_cam_mask
#from show_pictures import show_pictures
#from gradcam import GradCam
import random
#from wide_resnet import WideResNet
import argparse
import torchvision
import torch.utils.data as Data
import math
import sklearn.metrics as sk
#rom matplotlib import pyplot as plt
from resnet import ResNet34, ResNet18
import time
from resnet_DuBN import ResNet18_DuBN
def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    #print(y_score)
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    #print(recall)
    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)
    return auroc, aupr, fpr
def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
def test_acc(net,closerloader):
    print('test acc')
    net.eval()
    correct_know = 0
    sum_know = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #inputs = transform(inputs)
            batchnum = len(targets)
            sum_know = sum_know + batchnum
            logits = net(inputs.float())
            max_score_every_class, cls_pred = logits.max(dim=1)
            for i in range(batchnum):
                if cls_pred[i] == targets[i]:
                    correct_know = correct_know + 1
    acc_know = correct_know / sum_know
    print(acc_know)
def test(net,test_loader_in,test_loader_out):
    net.apply(lambda m: setattr(m, 'route', 'M'))
    net.eval()
    score_in = []
    score_out = []
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)

    with torch.no_grad():
        for batch_idx, (images,labels) in enumerate(test_loader_in):
            
            images = images.to(device)
            z = net(images.float())
            p = F.softmax(z, dim=1)
            
            max_score, cls_pred = p.max(dim=1) 
            #print(max_score)
            smax_yes = to_np(max_score)
            score_in.append(smax_yes)
        in_score = concat(score_in)[:len(test_loader_in.dataset)].copy()

        for batch_idx, (images,labels) in enumerate(test_loader_out):
            images = images.to(device)
            z = net(images.float())
            p = F.softmax(z, dim=1)
            
            max_score, cls_pred = p.max(dim=1) 

            smax_yes = to_np(max_score)
            score_out.append(smax_yes)
        out_score = concat(score_out)[:len(test_loader_out.dataset)].copy()

    auroc, aupr, fpr = get_measures(in_score, out_score)
    print(auroc)
    print(fpr)
    return auroc, fpr
'''
def funciton(x):
    return -1/90 * x + 250/90
'''

def funciton1(x, funciton_max_mask, funciton_temp):
    return 1 - math.exp((x - funciton_max_mask)/funciton_temp)

def funciton2(x, funciton_max_mask, funciton_temp):
    if x < 170:
        return 1
    elif x < 190:
        return 0.9
    elif x < 210:
        return 0.7
    elif x < 230:
        return 0.5
    elif x < 250:
        return 0.1
    else:
        return 0
         

'''
def funciton(x, lamda):
    return math.exp(-lamda * (x -155))
'''
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        #loss = -(F.softmax(target_probs, dim = 1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
    
class SmoothCrossEntropy_mixup(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(SmoothCrossEntropy_mixup, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels, labels_b, lam):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes

        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + self.alpha * (F.one_hot(labels, num_classes = num_classes) * lam + F.one_hot(labels_b, num_classes = num_classes) * (1-lam))
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        #loss = -(F.softmax(target_probs, dim = 1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--l1', type=float, default='0.2')
    parser.add_argument('--l2', type=float, default='1')
    parser.add_argument('--l3', type=float, default='0')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--max_epoch', type=int, default='300')
    parser.add_argument('--seed', type=int, default='30')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--lr', type=float, default='0.1')

    parser.add_argument('--alpha1', type=float, default='50')
    parser.add_argument('--beta1', type=float, default='20')
    
    parser.add_argument('--alpha2', type=float, default='0.5')
    parser.add_argument('--beta2', type=float, default='0.5')
    
    parser.add_argument('--alpha3', type=float, default='0.33')
    parser.add_argument('--beta3', type=float, default='3')

    parser.add_argument('--mixup_alpha', type=float, default='10')
    parser.add_argument('--mixup_beta', type=float, default='10')

    parser.add_argument('--min_mask', type=int, default='0')
    parser.add_argument('--gap', type=int, default='255')
    parser.add_argument('--funciton_max_mask', type=int, default='255')
    parser.add_argument('--funciton_temp', type=int, default='10')
    parser.add_argument('--funciton_choice', type=int, default='1')
    parser.add_argument('--envir', type=int, default='3')#post-hoc mask most important z
    
    args = parser.parse_args()
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed) 
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    l1 = args.l1
    l2 = args.l2 
    l3 = args.l3
    #for mask1 in [50, 130, 150, 170, 190, 210]:
    
    if args.envir == 0: #local
        base_root_data = 'G:/dataset/'
        base_root_model = 'E:/pretrained model/pretrained osr/'
        base_root_result = 'E:/pretrained model/'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        print(torch.cuda.get_device_name())
    
    elif args.envir == 1: #group
        base_root_data = '/home/common/niejun/dataset/'
        base_root_model = '/home/common/niejun/pretrained_model/pretrained_osr/'
        base_root_result = '/home/common/niejun/result/'
        if args.device == 0:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cuda:1")
        print("Using {} device".format(device))
        print(torch.cuda.get_device_name())
    
    elif args.envir == 3: #school
        base_root_data = '/gdata/niejun/dataset/'
        base_root_model = '/gdata/niejun/pretrained_model/pretrained_osr/'
        base_root_result = '/gdata/niejun/result/'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        print(torch.cuda.get_device_name())    
    
    if args.funciton_choice == 1:
        funciton = funciton1
    else:
        funciton = funciton2


    transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    
    transform_test = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    
    base_net = ResNet18()
    base_net.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    base_net.fc = torch.nn.Linear(512, 10) # 
    base_net.load_state_dict(torch.load(base_root_model + 'ERM_model_osr_cifar10_resnet18.pth'))
    #base_net.load_state_dict(torch.load(base_root_model + 'ERM_model_osr_cifar10_resnet18_300.pth'))
    base_net = base_net.to(device)
    base_net = base_net.eval()
    
    cifar_train = datasets.CIFAR10(root= base_root_data + 'cifar', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)
    
    cifar_test = datasets.CIFAR10(root=base_root_data + 'cifar', train=False, download=True, transform=transform_test)
    test_loader_in = torch.utils.data.DataLoader(cifar_test, batch_size=200, shuffle=False)

    testsetout = datasets.ImageFolder(root = base_root_data  +'dtd/images', transform=transform_test)
    test_loader_out1 = torch.utils.data.DataLoader(testsetout, batch_size=200,
                                            shuffle=False)

    test_dataset_svhn = torchvision.datasets.SVHN(
    root = base_root_data  + 'svhn',
    split='test',
    download=False,
    transform=transform_test
    )
    
    test_loader_out2 = Data.DataLoader(
        dataset=test_dataset_svhn,
        shuffle=False,
        batch_size=200
    )

    isun_data       = datasets.ImageFolder(root= base_root_data  + 'isun/iSUN' ,          transform=transform_test)
    test_loader_out3    = torch.utils.data.DataLoader(isun_data,       batch_size=200, shuffle=False)
    
    place_data = datasets.ImageFolder(root= base_root_data  + "place365",          transform=transform_test)
    test_loader_out4    = torch.utils.data.DataLoader(place_data,       batch_size=200, shuffle=False)
    
    lsun_data = datasets.ImageFolder(root= base_root_data  + "LSUN/LSUN",          transform=transform_test)
    test_loader_out5    = torch.utils.data.DataLoader(lsun_data,       batch_size=200, shuffle=False)
    
    '''
    net = ResNet18()
    net.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    net.fc = torch.nn.Linear(512, 10) #
    #net.load_state_dict(torch.load(base_root_model  + 'ERM_model_osr_cifar10_resnet18.pth'))
    #net.load_state_dict(torch.load(base_root_model  + 'ERM_model_osr_cifar10_resnet18_300.pth'))
    net = net.to(device)
    '''
    net = ResNet18_DuBN()
    net = net.to(device)

    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    criterion = torch.nn.CrossEntropyLoss()
    max_epoch = args.max_epoch
    learning_rate = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max_epoch)
    
    
    test_acc(net,test_loader_in)
    '''
    auroc1, fpr1 = test(net,test_loader_in,test_loader_out1)
    auroc2, fpr2 = test(net,test_loader_in,test_loader_out2)
    auroc3, fpr3 = test(net,test_loader_in,test_loader_out3)
    auroc4, fpr4 = test(net,test_loader_in,test_loader_out4)
    auroc5, fpr5 = test(net,test_loader_in,test_loader_out5)
    '''
    #net.apply(lambda m: setattr(m, 'route', 'M'))
    for epoch in range(max_epoch):
        print(epoch)
        for idx, (inputs, class_l) in enumerate(train_loader):
            
            
            r1 = np.random.beta(args.alpha1, args.beta1) * args.gap + args.min_mask
            r2 = np.random.beta(args.alpha2, args.beta2) * args.gap + args.min_mask
            r3 = np.random.beta(args.alpha3, args.beta3) * args.gap + args.min_mask
            
            inputs, class_l = inputs.to(device), class_l.to(device)
            
            mask_1 = np.ones([inputs.shape[0], 3, 32 ,32], dtype = int) #128 3 224 224
            mask_1 = torch.from_numpy(mask_1).to(device)
            mask_2 = np.ones([inputs.shape[0], 3, 32 ,32], dtype = int) #128 3 224 224
            mask_2 = torch.from_numpy(mask_2).to(device)
            mask_3 = np.ones([inputs.shape[0], 3, 32 ,32], dtype = int) #128 3 224 224
            mask_3 = torch.from_numpy(mask_3).to(device)
            
    
            net.train()
            net.zero_grad()
            optimizer.zero_grad()   #mask 加上mixup
            
            with torch.no_grad():
                    mask_single_1, mask_single_2, mask_single_3, _ = get_cam_mask(inputs.float(), class_l, base_net, t1 = r1, t2 = r2, t3 = r3)
                    mask_single_1 = torch.from_numpy(mask_single_1).cuda()
                    mask_single_2 = torch.from_numpy(mask_single_2).cuda()
                    mask_single_3 = torch.from_numpy(mask_single_3).cuda()
            mask_1[:,0,:,:] = mask_single_1[:,:,:]
            mask_1[:,1,:,:] = mask_single_1[:,:,:]
            mask_1[:,2,:,:] = mask_single_1[:,:,:]

            mask_2[:,0,:,:] = mask_single_2[:,:,:]
            mask_2[:,1,:,:] = mask_single_2[:,:,:]
            mask_2[:,2,:,:] = mask_single_2[:,:,:]
            
            mask_3[:,0,:,:] = mask_single_3[:,:,:]
            mask_3[:,1,:,:] = mask_single_3[:,:,:]
            mask_3[:,2,:,:] = mask_single_3[:,:,:]

            
            inputs1 = inputs * mask_1 

            smooth_loss1 = SmoothCrossEntropy(funciton(r1, args.funciton_max_mask, args.funciton_temp))
            #smooth_loss_hard = SmoothCrossEntropy(1)
            

            net.apply(lambda m: setattr(m, 'route', 'M'))#source and masked
            logits = net(inputs.float())
            class_loss = criterion(logits, class_l.long())
            
            logits1 = net(inputs1.float())
            class_loss1 = smooth_loss1(logits1, class_l) * l1
            #class_loss1 = smooth_loss_hard(logits1, class_l) * l1
            
            
            net.apply(lambda m: setattr(m, 'route', 'A'))#mixup
            mixup_x, part_y_a, part_y_b, lam = mixup_data(inputs, class_l, args.mixup_alpha, args.mixup_beta)
            inputs2 = inputs * mask_2  + mixup_x * (1-mask_2)
            smooth_loss2 = SmoothCrossEntropy_mixup(funciton(r2, args.funciton_max_mask, args.funciton_temp))
            logits2 = net(inputs2.float())
            class_loss2 = smooth_loss2(logits2, class_l, part_y_b, lam) * l2
            #class_loss2 = smooth_loss_hard(logits2, class_l) * l2
            

            '''
            logits_mixup = net(mixup_x)
            loss_mixup = regmixup_criterion(F.cross_entropy, logits_mixup, part_y_a, part_y_b, lam) * l2
            '''

            
            loss = class_loss  + class_loss1 + class_loss2


            loss.backward()
            optimizer.step()
        scheduler.step()
          
    auroc1, fpr1 = test(net,test_loader_in,test_loader_out1)
    auroc2, fpr2 = test(net,test_loader_in,test_loader_out2)
    auroc3, fpr3 = test(net,test_loader_in,test_loader_out3)
    auroc4, fpr4 = test(net,test_loader_in,test_loader_out4)
    auroc5, fpr5 = test(net,test_loader_in,test_loader_out5)
    avg_auroc = (auroc1+auroc2+auroc3+auroc4+auroc5)/5
    avg_fpr = (fpr1+ fpr2 + fpr3 + fpr4 + fpr5)/5
    print('avg auroc')
    print(avg_auroc)
    print('avg fpr')
    print(avg_fpr)
    print('**********')
    test_acc(net,test_loader_in)
    #torch.save(net.state_dict(), base_root_model + 'voso_reg_model_osr_cifar10_resnet18_s100.pth')
    
    

        
