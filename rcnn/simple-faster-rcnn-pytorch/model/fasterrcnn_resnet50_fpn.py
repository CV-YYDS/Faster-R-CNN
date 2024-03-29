import torch as t
from torch import nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

def set_bn_fix(m):
    classname=m.__class__.__name__
    if classname.find("BatchNorm")!=-1:
        for p in m.parameters():p.require_grad=False

def decom_resnet50():

    if opt.caffe_pretrain:
        model = resnet_fpn_backbone(backbone_name="resnet50",pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = resnet_fpn_backbone(backbone_name="resnet50",pretrained=False)

    # freeze top conv and bn layers
    for p in model.conv1.parameters(): p.requires_grad = False
    for p in model.layer1.parameters(): p.requires_grad = False
    model.apply(set_bn_fix)

    # resnet.layer0 to resnet.layer3 for extractor
    features_extractor = nn.Sequential(model.conv1, model.bn1,model.relu,
      model.maxpool,model.layer1,model.layer2,model.layer3)

    # layer4 for classifier
    features_classifier = nn.Sequential(model.layer4)

    return features_extractor, features_classifier


class FasterRCNNRESNET50FPN(FasterRCNN):
    """Faster R-CNN based on RESNET101.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.
    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
    """

    feat_stride = 16  # downsample 16x for output of conv5 in resnet101

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_resnet50()

        rpn = RegionProposalNetwork(
            1024, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = RESNET50RoIHead(
            n_class=n_fg_class + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNRESNET50FPN, self).__init__(
            extractor,
            rpn,
            head,
        )


class RESNET50RoIHead(nn.Module):
    """Faster R-CNN Head for RESNET101 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): layer4 feature Linear ported from resnet101
    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(RESNET50RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x, rois, roi_indices):
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        feature = self.classifier(pool)
        feature = self.avgpool(feature)

        feature = feature.view(feature.size(0), -1)
        roi_cls_locs = self.cls_loc(feature)
        roi_scores = self.score(feature)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()