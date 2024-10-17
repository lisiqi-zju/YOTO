_base_ = (
    '../third_party/mmyolo/configs/yolov8/'
    'yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(
    imports=['yolo_world','yoto'],
    allow_failed_imports=False)


num_classes = 1
num_training_classes = 1
max_epochs = 4  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05    
train_batch_size_per_gpu = 4
load_from = 'pretrained_models/yolo_world_v2_s_vlpan_bn_2e-4_80e_8gpus_mask-refine_finetune_coco_ep80-492dc329.pth'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False
mixup_prob = 0.15
copypaste_prob = 0.3



# model settings
model = dict(
    type='YOTODetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOTODataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            frozen_stages=4),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              ),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes,
                                    freeze_all=True)),
    backbone_teacher=dict(
        type='MultiModalYOLOBackbone',
        image_model=dict(
            type='YOLOv8CSPDarknet',
            arch='P5',
            last_stage_out_channels=_base_.last_stage_out_channels,
            deepen_factor=_base_.deepen_factor,
            widen_factor=_base_.widen_factor,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            frozen_stages=4),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck_teacher=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
            in_channels=[256, 512, _base_.last_stage_out_channels,],
            out_channels=[256, 512, _base_.last_stage_out_channels,],
            num_heads=neck_num_heads,
            block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              ),
    bbox_head_teacher=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    in_channels=[256, 512,  _base_.last_stage_out_channels],
                                    num_classes=num_training_classes,
                                    freeze_all=True)),

    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    coco_path=load_from
    )

    

use_mask2refine = True
pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='LoadTDODAnnotations',
        with_bbox=True,
        with_mask=True,
        mask2bbox=use_mask2refine)
]

mosaic_affine_transform = [
    dict(
        type='TDODMultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='TDODRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine)
]


train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='TDODMultiModalMixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform,
                       *mosaic_affine_transform]),
    dict(type='RemoveDataElement', keys=['gt_masks']),
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts', 'gt_coco_labels','coco_texts'))
]


dataset1 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_1_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist1.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset2 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_2_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist2.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset3 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_3_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist3.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset4 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_4_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist4.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset5 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_5_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist5.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset6 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_6_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist6.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset7 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_7_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist7.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset8 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_8_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist8.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset9 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_9_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist9.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset10 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_10_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist10.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset11 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_11_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist11.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)
dataset12 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_12_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist10.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset13 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_13_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist13.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

dataset14 = dict(
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
        data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_14_train.json',
        data_prefix=dict(img='train2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist14.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

# coco_train_dataset=dict(_delete_=True,type='ConcatDataset', datasets=[dataset1,dataset2,dataset3,dataset4,
#                                                                       dataset5,dataset6,dataset7,dataset8,
#                                                                       dataset9,dataset10,dataset11,dataset12,
#                                                                        dataset13,dataset14])
coco_train_dataset=dict(_delete_=True,type='ConcatDataset', datasets=[dataset4,dataset6])

train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolot_collate'),
    dataset=coco_train_dataset)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param', 'texts','gt_coco_labels'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='YTModalDataset',
    dataset=dict(
        type='TDODDataset',
         data_root='data/data/images',
        ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_4_test.json',
        data_prefix=dict(img='val2014/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/toist4.json',
    coco_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline)


val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=-1,
        save_best=None,
        interval=save_epoch_intervals))


train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)])
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.001),
                    # 'backbone':dict(lr_mult=0),
                    #  'neck':dict(lr_mult=0),
                    #  'bbox_head':dict(lr_mult=0),
                     
                     'logit_scale': dict(weight_decay=0.0),
                    #  'backbone_teacher':dict(lr_mult=0,weight_decay=0.0),
                    #  'neck_teacher':dict(lr_mult=0,weight_decay=0.0),
                    #  'bbox_head_teacher':dict(lr_mult=0,weight_decay=0.0)
                     
                     }),
    constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='SimpleAccuracy',
    proposal_nums=(100, 1, 10),
    ann_file='/data/lisq2309/YOTO/data/data/coco-tasks/annotations/task_4_test.json',
    metric='bbox')

test_evaluator=val_evaluator

find_unused_parameters = True

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
                  save_dir='video_outputs')