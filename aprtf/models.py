# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

# general
import math
import torch

# architectures
import torchvision.models.detection as td


class ModelBuilder:
    @staticmethod
    def build_detector(args, num_classes, weights):
        pretrained = (len(weights) > 0)
        if args.arch == 'fasterrcnn_resnet50_fpn':
            if pretrained:
                detector = td.fasterrcnn_resnet50_fpn_v2(weights=None) 
            else:
                detector = td.fasterrcnn_resnet50_fpn_v2(weights=td.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT) 
            # replace the pre-trained head with a new one
            in_features = detector.roi_heads.box_predictor.cls_score.in_features
            detector.roi_heads.box_predictor = td.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        elif args.arch == 'retinanet_resnet50_fpn':
            if pretrained:
                detector = td.retinanet_resnet50_fpn_v2(weights=None)  
            else:
                detector = td.retinanet_resnet50_fpn_v2(weights=td.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
            # replace the pre-trained head with a new one
            in_features = detector.head.classification_head.cls_logits.in_channels
            num_anchors = detector.head.classification_head.num_anchors
            # only cls logits at end depends on class size
            detector.head.classification_head.num_classes = num_classes
            cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
            torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
            torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code
            detector.head.classification_head.cls_logits = cls_logits
        else:
            raise Exception('Architecture undefined!')

        if pretrained:
            print('Loading weights for detector')
            detector.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return detector


class OptimizerBuilder:
    @staticmethod
    def build_optimizer(args, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if args.optim == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr,
                                        momentum=args.momentum, 
                                        weight_decay=args.weight_decay)
        else:
            raise Exception('Optimizer undefined!')
        return optimizer


class LRScheduleBuilder:
    @staticmethod
    def build_scheduler(args, optimizer):
        if args.schedule == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=args.step_size,
                                                           gamma=args.gamma)  
        else:
            raise Exception('LR Scheduler undefined!')
        return lr_scheduler