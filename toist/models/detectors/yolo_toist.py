from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .kmeans import kmeans
from .kmeans import kmeans_predict



@MODELS.register_module()
class YOLOTOIST(BaseDetector):

    def __init__(self,
                backbone: ConfigType,
                neck: ConfigType,
                bbox_head: ConfigType,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                use_syncbn: bool = True,
                mm_neck: bool = False,
                num_train_classes=80,
                num_test_classes=80) -> None:
  
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                            local_metadata: dict, strict: bool,
                            missing_keys: Union[List[str], str],
                            unexpected_keys: Union[List[str], str],
                            error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results


    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class YTTeacher(BaseDetector):

    def __init__(self,
                backbone: ConfigType,
                neck: ConfigType,
                bbox_head: ConfigType,
                coco_path,
                train_cfg: OptConfigType = None,
                test_cfg: OptConfigType = None,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
                use_syncbn: bool = True,
                mm_neck: bool = False,
                num_train_classes=80,
                num_test_classes=80) -> None:
  
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # backbone['frozen_stages']=4
        # neck['freeze_all']=True
        # bbox_head['head_module']['freeze_all']=True

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)



        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""

        self.bbox_head.num_classes = self.num_train_classes
        coco_batch_data_samples = batch_data_samples
        for i in range(len(coco_batch_data_samples['coco_texts'])):
            coco_batch_data_samples['texts'][i] = ['chair']
        # coco_batch_data_samples['bboxes_labels']=batch_data_samples['coco_bboxes_labels']
        # coco_batch_data_samples['texts']=batch_data_samples['coco_texts']
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 coco_batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, coco_batch_data_samples)
        # losses['loss_cls'] *=0
        # losses['loss_bbox'] *=0
        # losses['loss_dfl'] *=0
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        # coco_batch_data_samples = batch_data_samples
        # coco_batch_data_samples['bboxes_labels']=batch_data_samples['coco_bboxes_labels']
        # coco_batch_data_samples['texts']=batch_data_samples['coco_texts']

        batch_data_samples[0].set_metainfo(dict(texts = ['chair']))
        # for i in range(len(batch_data_samples['coco_texts'])):
        #     batch_data_samples['texts'][i] = ['chair']

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results


    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats

import copy
import torch.nn.functional as F
import time  
from torch.utils.tensorboard import SummaryWriter   



coco_categories = {  
    "1": "person", 
    "2": "bicycle",  
    "3": "car",  
    "4": "motorcycle",  
    "5": "airplane",  
    "6": "bus",  
    "7": "train",  
    "8": "truck",  
    "9": "boat",  
    "10": "traffic light",  
    "11": "fire hydrant",  
    "13": "stop sign",  
    "14": "parking meter",  
    "15": "bench",  
    "16": "bird",  
    "17": "cat",  
    "18": "dog",  
    "19": "horse",  
    "20": "sheep",  
    "21": "cow",  
    "22": "elephant",  
    "23": "bear",  
    "24": "zebra",  
    "25": "giraffe",  
    "27": "backpack",  
    "28": "umbrella",  
    "31": "handbag",  
    "32": "tie",  
    "33": "suitcase",  
    "34": "frisbee",  
    "35": "skis", 
    "36": "snowboard",  
    "37": "sports ball",  
    "38": "kite",  
    "39": "baseball bat",  
    "40": "baseball glove",  
    "41": "skateboard",  
    "42": "surfboard",  
    "43": "tennis racket",  
    "44": "bottle",  
    "46": "wine glass",  
    "47": "cup",  
    "48": "fork", 
    "49": "knife",  
    "50": "spoon",  
    "51": "bowl",  
    "52": "banana",  
    "53": "apple",  
    "54": "sandwich",  
    "55": "orange",  
    "56": "broccoli",  
    "57": "carrot",  
    "58": "hot dog",  
    "59": "pizza",  
    "60": "donut",  
    "61": "cake",  
    "62": "chair",  
    "63": "couch",  
    "64": "potted plant",  
    "65": "bed", 
    "67": "dining table",  
    "70": "toilet",  
    "72": "tv",  
    "73": "laptop",  
    "74": "mouse",  
    "75": "remote",  
    "76": "keyboard",  
    "77": "cell phone",  
    "78": "microwave",  
    "79": "oven",  
    "80": "toaster", 
    "81": "sink",  
    "82": "refrigerator",  
    "84": "book",  
    "85": "clock",  
    "86": "vase",  
    "87": "scissors",  
    "88": "teddy bear",  
    "89": "hair drier",  
    "90": "toothbrush"  
}

@MODELS.register_module()

class DistillModel(BaseDetector):

    def __init__(self,
            backbone: ConfigType,
            neck: ConfigType,
            bbox_head: ConfigType,
            backbone_teacher: ConfigType,
            neck_teacher: ConfigType,
            bbox_head_teacher: ConfigType,
            coco_path,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            use_syncbn: bool = True,
            mm_neck: bool = False,
            num_train_classes=1,
            num_test_classes=1,) -> None:
        
        self.coco_categories=coco_categories
        self.mm_neck = mm_neck
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)

        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)      
        self.bbox_head = MODELS.build(bbox_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # backbone['image_model']['frozen_stages']=4

        backbone_teacher=copy.deepcopy(backbone)
        neck_teacher=copy.deepcopy(neck)
        bbox_head_teacher=copy.deepcopy(bbox_head)
        
        self.backbone_teacher = MODELS.build(backbone_teacher)
        self.neck_teacher = MODELS.build(neck_teacher)
        self.bbox_head_teacher = MODELS.build(bbox_head_teacher)

        self.T=1
        self.predict_load_flag=False
        
        self.coco_path='pretrained_models/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth'
        # state_dict = torch.load(coco_path)

        # backbone_state_dict = {k.replace('backbone.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}  
        # self.backbone_teacher.load_state_dict(backbone_state_dict)

        # neck_state_dict = {k.replace('neck.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}  
        # self.neck_teacher.load_state_dict(neck_state_dict)

        # bbox_head_state_dict = {k.replace('bbox_head.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('bbox_head.')}  
        # self.bbox_head_teacher.load_state_dict(bbox_head_state_dict)

        # cp_path='/data/lisq2309/YT/41.3.pth'
        # state_dict = torch.load(cp_path)
        # backbone_state_dict = {k.replace('backbone.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}  
        # self.backbone.load_state_dict(backbone_state_dict)

        # neck_state_dict = {k.replace('neck.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}  
        # self.neck.load_state_dict(neck_state_dict)

        # bbox_head_state_dict = {k.replace('bbox_head.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('bbox_head.')}  
        # self.bbox_head.load_state_dict(bbox_head_state_dict)

        

        self.memory_bank0=[]
        self.memory_bank1=[]
        self.memory_bank2=[]

    def get_coco_text(self, bboxes_labels,batch_size):
        
        coco_text=[]
        for i in range(batch_size):
            for bboxes_label in bboxes_labels:
                if bboxes_label[0]!=i:
                    continue
                coco_idx=bboxes_label[1]
                text=self.coco_categories[str(int(coco_idx.item()))]
                coco_text.append(text)
            
        coco_text = list(set(coco_text)) 

        coco_texts=[]
        for i in range(batch_size):
            coco_texts.append(coco_text)

        return coco_texts


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_train_classes

        state_dict = torch.load(self.coco_path)

        backbone_state_dict = {k.replace('backbone.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}  
        self.backbone_teacher.load_state_dict(backbone_state_dict)

        neck_state_dict = {k.replace('neck.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}  
        self.neck_teacher.load_state_dict(neck_state_dict)

        bbox_head_state_dict = {k.replace('bbox_head.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('bbox_head.')}  
        self.bbox_head_teacher.load_state_dict(bbox_head_state_dict)


        coco_batch_data_samples = copy.deepcopy(batch_data_samples)

        coco_texts=self.get_coco_text(batch_data_samples['coco_bboxes_labels'],len(batch_data_samples['texts']))



        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        
        # img_feats_clone=(img_feats[0].clone(),img_feats[1].clone(),img_feats[2].clone())
        # txt_feats_clone=txt_feats.clone()

        # img_feats_clone=img_feats
        # txt_feats_clone=txt_feats


        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)


        if len(coco_texts[0])!=0:
            coco_texts_list=[]
            for text in coco_texts[0]:
                coco_texts_list.append([text])
            coco_batch_data_samples['texts'] = coco_texts

        img_feats_teacher, txt_feats_teacher = self.teacher_extract_feat(batch_inputs,
                                                coco_batch_data_samples)

        
        ##############################################

        
        # if len(self.memory_bank0)==2:
        # tensor_list=[a.unsqueeze(0) for a in self.memory_bank0 ]
        # concatenated_tensor = torch.cat(tensor_list, dim=0) 

        memory_bank=torch.cat((img_feats_teacher[0].unsqueeze(0),img_feats[0].unsqueeze(0)),dim=0)

        _,new_cluster_centers=kmeans(memory_bank,None,1,device=memory_bank.device)
        cluster_ids_y = kmeans_predict(memory_bank, new_cluster_centers, 'euclidean', device=memory_bank.device)
        cluster_center_choice = cluster_ids_y[0]
        cluster_center_feature0 = new_cluster_centers[cluster_center_choice]

        memory_bank=torch.cat((img_feats_teacher[1].unsqueeze(0),img_feats[1].unsqueeze(0)),dim=0)

        _,new_cluster_centers=kmeans(memory_bank,None,1,device=memory_bank.device)
        cluster_ids_y = kmeans_predict(memory_bank, new_cluster_centers, 'euclidean', device=memory_bank.device)
        cluster_center_choice = cluster_ids_y[0]
        cluster_center_feature1 = new_cluster_centers[cluster_center_choice]

        memory_bank=torch.cat((img_feats_teacher[2].unsqueeze(0),img_feats[2].unsqueeze(0)),dim=0)

        _,new_cluster_centers=kmeans(memory_bank,None,1,device=memory_bank.device)
        cluster_ids_y = kmeans_predict(memory_bank, new_cluster_centers, 'euclidean', device=memory_bank.device)
        cluster_center_choice = cluster_ids_y[0]
        cluster_center_feature2 = new_cluster_centers[cluster_center_choice]


        # tensor_list=[a.unsqueeze(0) for a in self.memory_bank1 ]
        # concatenated_tensor = torch.cat(tensor_list, dim=0) 
        # _,new_cluster_centers=kmeans(concatenated_tensor,None,1,device=concatenated_tensor.device)
        # cluster_ids_y = kmeans_predict(concatenated_tensor, new_cluster_centers, 'euclidean', device=concatenated_tensor.device)
        # cluster_center_choice = cluster_ids_y[0]
        # cluster_center_feature1 = new_cluster_centers[cluster_center_choice]

        # tensor_list=[a.unsqueeze(0) for a in self.memory_bank2 ]
        # concatenated_tensor = torch.cat(tensor_list, dim=0) 
        # _,new_cluster_centers=kmeans(concatenated_tensor,None,1,device=concatenated_tensor.device)
        # cluster_ids_y = kmeans_predict(concatenated_tensor, new_cluster_centers, 'euclidean', device=concatenated_tensor.device)
        # cluster_center_choice = cluster_ids_y[0]
        # cluster_center_feature2 = new_cluster_centers[cluster_center_choice]

        cluster_feats=(cluster_center_feature0,cluster_center_feature1,cluster_center_feature2)
        losses = self.bbox_head.loss(cluster_feats, txt_feats, batch_data_samples)

            # for _,key in enumerate(losses):
            #     losses[key]+=cluster_loss[key]
        # else:
        # losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)

        ################################################
        
        




        # for i,text in enumerate(coco_texts[0]):
        #     for j in range(len(coco_batch_data_samples['coco_texts'])):
        #         coco_batch_data_samples['texts'][j] = [text]

        #     img_feats_teacher, txt_feats_teacher = self.teacher_extract_feat(batch_inputs,
        #                                         coco_batch_data_samples)
        #     if  i==0:
        #         kl_loss=self.distill_loss(img_feats[0],img_feats_teacher[0])
        #     else:
        #         kl_loss+=self.distill_loss(img_feats[0],img_feats_teacher[0])
        #     kl_loss+=self.distill_loss(img_feats[1],img_feats_teacher[1])
        #     kl_loss+=self.distill_loss(img_feats[2],img_feats_teacher[2])
            
        
        
        # losses_teacher = self.bbox_head.loss(img_feats_teacher, txt_feats_teacher, coco_batch_data_samples)





        kl_loss=self.distill_loss(img_feats[0],img_feats_teacher[0])
        kl_loss+=self.distill_loss(img_feats[1],img_feats_teacher[1])
        kl_loss+=self.distill_loss(img_feats[2],img_feats_teacher[2])

        if len(coco_texts[0])!=0:
            losses['kd_loss']=kl_loss*self.T
        self.T-=1/18000
        if self.T<0:
            self.T=0
        
        return losses

    def distill_loss(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        self.tau=1
        self.loss_weight=100
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / self.tau, dim=1)

        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(preds_T.view(-1, W * H) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(preds_S.view(-1, W * H) / self.tau)) * (
                             self.tau**2)

        loss = self.loss_weight * loss / (C * N)

        return loss
    
    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        if  self.predict_load_flag ==False:
            cp_path='/data/lisq2309/YT/41.3.pth'
            # cp_path='/data/lisq2309/YT/work_dirs/distill/epoch_1.pth'
            state_dict = torch.load(cp_path)
            backbone_state_dict = {k.replace('backbone.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}  
            self.backbone.load_state_dict(backbone_state_dict)

            neck_state_dict = {k.replace('neck.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}  
            self.neck.load_state_dict(neck_state_dict)

            bbox_head_state_dict = {k.replace('bbox_head.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('bbox_head.')}  
            self.bbox_head.load_state_dict(bbox_head_state_dict)
            self.predict_load_flag= True

        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        if  self.predict_load_flag ==False:
            cp_path='/data/lisq2309/YT/41.3.pth'
            # cp_path='/data/lisq2309/YT/work_dirs/distill/epoch_1.pth'
            state_dict = torch.load(cp_path)
            backbone_state_dict = {k.replace('backbone.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('backbone.')}  
            self.backbone.load_state_dict(backbone_state_dict)

            neck_state_dict = {k.replace('neck.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('neck.')}  
            self.neck.load_state_dict(neck_state_dict)

            bbox_head_state_dict = {k.replace('bbox_head.', '') : v for k, v in state_dict['state_dict'].items() if k.startswith('bbox_head.')}  
            self.bbox_head.load_state_dict(bbox_head_state_dict)
            self.predict_load_flag= True

        # batch_data_samples[0].set_metainfo(dict(texts = ['chair']))


        # start_time = time.time()  


        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
       

        self.bbox_head.num_classes = txt_feats[0].shape[0]
        
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)
        

        

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
    

    def teacher_extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck_teacher(img_feats, txt_feats)
            else:
                img_feats = self.neck_teacher(img_feats)
        return img_feats, txt_feats


    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results



# def memory_cluster(feature_to_cluster):
#     memory_feature=feature_to_cluster
#     new_cluster_centers=kmeans(memory_feature,None,3)
#     cluster_ids_y = kmeans_predict(
#             feature_to_cluster.reshape(1,-1), new_cluster_centers, 'euclidean', device=new_cluster_centers.device)
    
#     cluster_center_choice = cluster_ids_y[0]
#     cluster_center_feature = new_cluster_centers[cluster_center_choice]

#     return cluster_center_choice, cluster_center_feature