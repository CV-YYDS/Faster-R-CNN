import sys
import torch
import numpy as np
from torchvision.models import vgg16, resnet50
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from roi_align.functions.roi_align import RoIAlignFunction
from utils.config import opt
from utils import array_tool
from torch.nn import functional
from model.fpn import FPN

def decom_vgg16():
    model = vgg16(not opt.load_path)

    C2 = list(model.features)[:16]
    C3 = list(model.features)[16:23]
    C4 = list(model.features)[23:30]

    classifier = model.classifier  
    classifier = list(classifier)
    del classifier[6] 
    if not opt.use_dropout:
        del classifier[5]
        del classifier[2]
    classifier = torch.nn.Sequential(*classifier)

    return torch.nn.Sequential(*C2), torch.nn.Sequential(*C3), torch.nn.Sequential(*C4), classifier


class FasterRCNNVGG16FPN(FasterRCNN):

    def __init__(self,
                 n_fg_class= opt.class_num,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16],
                 feat_stride=[4, 8, 16, 32]):
        C2, C3, C4, classifier = decom_vgg16()

        fpn = FPN(
            out_channels=512
        )

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            feat_stride=[4, 8, 16, 32],
            classifier=classifier,
        )

        super(FasterRCNNVGG16FPN, self).__init__(
            C2, C3, C4,
            fpn,
            rpn,
            head,
        )

class VGG16RoIHead(torch.nn.Module):
    def __init__(self, n_class, roi_size, feat_stride, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = torch.nn.Linear(4096, n_class * 4)
        self.score = torch.nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.feat_stride = feat_stride
        self.spatial_scale = [1. / i for i in feat_stride]

    def forward(self, features_maps, rois, roi_indices):
        roi_indices = array_tool.totensor(roi_indices).float()
        rois = array_tool.totensor(rois).float()
        roi_level = self._PyramidRoI_Feat(rois)
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  
        indices_and_rois = xy_indices_and_rois.contiguous()  

        roi_pool_feats = []
        roi_to_levels = []

        for i, l in enumerate(range(2, 5)):
            if (roi_level == l).sum() == 0:
                continue
            idx_l = (roi_level == l).nonzero()
            roi_to_levels.append(idx_l)

            keep_indices_and_rois = indices_and_rois[idx_l]
            keep_indices_and_rois = keep_indices_and_rois.view(-1, 5)
            #roi_pooling = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale[i])
            roi_align = RoIAlignFunction(self.roi_size, self.roi_size, self.spatial_scale[i])
            #pool = roi_pooling(features_maps[i], keep_indices_and_rois)   #通过roi_pooling
            pool = roi_align(features_maps[i], keep_indices_and_rois)
            roi_pool_feats.append(pool)
        roi_pool_feats = torch.cat(roi_pool_feats, 0)
        roi_to_levels = torch.cat(roi_to_levels, 0)
        roi_to_levels = roi_to_levels.squeeze()
        idx_sorted, order = torch.sort(roi_to_levels)
        roi_pool_feats = roi_pool_feats[order]

        pool = roi_pool_feats.view(roi_pool_feats.size(0), -1)

        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7) 
        roi_scores = self.score(fc7) 

        return roi_cls_locs, roi_scores

    def _PyramidRoI_Feat(self, rois):
        roi_h = rois[:, 2] - rois[:, 0] + 1
        roi_w = rois[:, 3] - rois[:, 1] + 1
        roi_level = torch.log(torch.sqrt(roi_h*roi_w)/224.0) /np.log(2)
        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 4] = 4
        return roi_level


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) 
    else:
        m.weight.data.normal_(mean, stddev)  
        m.bias.data.zero_()

def weights_init(m, mean, stddev, truncated=False):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



