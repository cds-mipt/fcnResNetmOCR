import torchvision

from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .base_oc_block import BaseOC_Module
from .pyramid_oc_block import Pyramid_OC_Module
from .asp_oc_block import ASP_OC_Module
from .backbone_utils import IntermediateLayerGetter


class ResNet_M(nn.Module):
    def __init__(self, backbone, inplanes, num_classes):
        super(ResNet_M, self).__init__()
        self.backbone = backbone
        self.cls = nn.Conv2d(inplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.cls(x)

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


def get_resnet34_layer3(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation = [False, False, False]

    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    return_layers = {'layer3': 'out'}
    inplanes = 256

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = ResNet_M(backbone, inplanes, num_classes)

    return model


class ResNet_Base_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Base_OC, self).__init__()
        self.backbone = backbone
        self.context = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            BaseOC_Module(in_channels=outplanes, 
                          out_channels=outplanes, 
                          key_channels=outplanes // 2, 
                          value_channels=outplanes // 2, 
                          dropout=0.05, 
                          sizes=([1]))
            )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # print('input: ', x.shape)
        x = self.backbone(x)['out']
        # print('backbone: ', x.shape)
        x = self.context(x)
        # print('context: ', x.shape)
        x = self.cls(x)
        # print('cls: ', x.shape)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        # print('output: ', x.shape)
        
        result = OrderedDict()
        result['out'] = x
        
        return result


class ResNet_Pyramid_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_Pyramid_OC, self).__init__()
        self.backbone = backbone
        self.context = Pyramid_OC_Module(in_channels=inplanes, 
                                         out_channels=outplanes, 
                                         dropout=0.05, 
                                         sizes=([1, 2, 3, 6]))
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        x = self.backbone(x)['out']
        x = self.context(x)
        x = self.cls(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        result = OrderedDict()
        result['out'] = x
        
        return result


class ResNet_ASP_OC(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet_ASP_OC, self).__init__()
        self.backbone = backbone
        self.context = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outplanes),
                ASP_OC_Module(outplanes, outplanes // 2)
                )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        
        x = self.backbone(x)['out']
        x = self.context(x)
        x = self.cls(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        result = OrderedDict()
        result['out'] = x
        
        return result


def get_resnet34_base_oc_layer2(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4

    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer2': 'out'}
    inplanes = 512 // inplanes_scale_factor
    outplanes = 256
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


def get_resnet34_base_oc_layer3(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation = [False, False, False]
    inplanes_scale_factor = 4
    
    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer3': 'out'}
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 512
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_Base_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


def get_resnet34_pyramid_oc_layer2(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation=[False, False, False]
    inplanes_scale_factor = 4
    
    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer2': 'out'}
    inplanes = 512 // inplanes_scale_factor
    outplanes = 128
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_Pyramid_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


def get_resnet34_pyramid_oc_layer3(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation=[False, False, False]
    inplanes_scale_factor = 4
    
    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer3': 'out'}
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 256
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_Pyramid_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


def get_resnet34_asp_oc_layer2(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation=[False, False, False]
    inplanes_scale_factor = 4
    
    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer2': 'out'}
    inplanes = 512 // inplanes_scale_factor
    outplanes = 256
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_ASP_OC(backbone, inplanes, outplanes, num_classes)
    
    return model


def get_resnet34_asp_oc_layer3(num_classes=66, pretrained_backbone=False):
    backbone_name = 'resnet34'
    replace_stride_with_dilation=[False, False, False]
    inplanes_scale_factor = 4
    
    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {'layer3': 'out'}
    inplanes = 1024 // inplanes_scale_factor
    outplanes = 512
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet_ASP_OC(backbone, inplanes, outplanes, num_classes)
    
    return model

class ResNet18_model(nn.Module):
    def __init__(self, backbone, inplanes, outplanes, num_classes):
        super(ResNet18_model, self).__init__()
        self.backbone = backbone
        self.context = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            BaseOC_Module(in_channels=outplanes,
                          out_channels=outplanes,
                          key_channels=outplanes // 2,
                          value_channels=outplanes // 2,
                          dropout=0.05,
                          sizes=([1]))
        )
        self.cls = nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        x = self.backbone(x)['out']
        x = self.context(x)
        x = self.cls(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


def get_resnet18_layer3_ocr(num_classes=10, pretrained_backbone=False):
    backbone_name = 'resnet18'
    replace_stride_with_dilation = [False, False, False]

    backbone = torchvision.models.resnet.__dict__[backbone_name](pretrained=pretrained_backbone,
                                                                 replace_stride_with_dilation=replace_stride_with_dilation)

    return_layers = {'layer3': 'out'}
    inplanes = 64
    outplanes = 64
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = ResNet18_model(backbone, inplanes, outplanes, num_classes)

    return model
