import io
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import torch
def get_cam_mask(x, class_l, net, t1, t2, t3, cam_num = 3):
    # a batch(x,y),the classifier network resnet18
    # return mask where mask = 1 if cam of x <200 else mask = 0
    # 让网络关注更广泛的区域
    finalconv_name = 'layer4'
    net.eval()
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
    handle = net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) 
    
    def returnCAM(feature_conv, weight_softmax, idx, class_l):
        size_upsample = (32, 32)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for i in range(bz):
            cam = weight_softmax[idx[i][0]].dot(feature_conv[i,None,:,:].reshape((nc, h*w)))
            #cam = weight_softmax[class_l[i]].dot(feature_conv[i,None,:,:].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / (np.max(cam)+1e-5)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return np.array(output_cam)
    logit = net(x)
    logit = logit.cpu()
    h_x = F.softmax(logit, dim=-1).data.squeeze()
    probs, idx = h_x.sort(1, True)
    probs = probs.numpy()
    idx = idx.numpy()
    #print(class_l)
    #print(idx)
    #print('**************')
    CAMs = returnCAM(features_blobs[0], weight_softmax, idx, class_l) #输出概率最高的类的cam
    _,_,height, width = x.shape
    mask_1 = np.ones(CAMs.shape)
    mask_1[np.where((CAMs >= t1))] = 0
    mask_2 = np.ones(CAMs.shape)
    mask_2[np.where((CAMs >= t2))] = 0
    mask_3 = np.ones(CAMs.shape)
    mask_3[np.where((CAMs >= t3))] = 0
    
    handle.remove()
    if cam_num == 1:
        return mask_1, CAMs
    else:
        return mask_1, mask_2, mask_3, CAMs
