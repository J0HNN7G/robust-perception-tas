# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

import torch

# architectures
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2 
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class ModelBuilder:

    @staticmethod
    def build_detector(arch='fasterrcnn_resnet50_fpn', num_classes=2, weights=''):
        pretrained = (len(weights) > 0)
        arch = arch.lower()
        if arch == 'fasterrcnn_resnet50_fpn':
            if pretrained:
                detector = fasterrcnn_resnet50_fpn_v2(weights=None) 
            else:
                detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights) 
            # replace the pre-trained head with a new one
            in_features = detector.roi_heads.box_predictor.cls_score.in_features
            detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    
        elif arch == 'retinanet_resnet50_fpn':
            if pretrained:
                detector = retinanet_resnet50_fpn_v2(weights=None)  
            else:
                detector = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights)
            # replace the pre-trained head with a new one
            in_features = detector.head.classification_head.cls_logits.in_channels
            num_anchors = detector.head.classification_head.num_anchors
            detector.head.classification_head = RetinaNetClassificationHead(in_features, num_classes, num_anchors)
        elif arch == 'maskrcnn_resnet50_fpn':
                if pretrained:
                    model = maskrcnn_resnet50_fpn_v2(weights=None)
                else:
                    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
                # replace the class head with a new one matching num classes
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                # replace the mask head with a new one matching num classes
                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                hidden_layer = 256
                model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
                detector = model
        else:
            raise Exception('Architecture undefined!')

        if pretrained:
            print('Loading weights for detector')
            detector.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            
        return detector