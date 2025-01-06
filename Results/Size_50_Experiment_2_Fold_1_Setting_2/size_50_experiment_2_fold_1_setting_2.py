dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=['geospatial_fm'])
num_frames = 6
img_size = 366
pretrained_weights_path = '/home/featurize/work/hls-foundation-os/Current_Pretrained_Prithvi_Weights/Prithvi_EO_V1_100M.pt'
num_layers = 6
patch_size = 61
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 20
eval_epoch_interval = 1
loss_weights_multi = [
    48.24197892603771, 1.3908071469704752, 13.91045298879203,
    0.16260287710088167, 24.915739196406832, 1.04194004835998,
    0.44759908979569946, 2.6068422990207574, 13.745187372306951,
    33.410956287558186, 3.143579631887296
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=[
        48.24197892603771, 1.3908071469704752, 13.91045298879203,
        0.16260287710088167, 24.915739196406832, 1.04194004835998,
        0.44759908979569946, 2.6068422990207574, 13.745187372306951,
        33.410956287558186, 3.143579631887296
    ],
    avg_non_ignore=True)
output_embed_dim = 4608
experiment = 'Size_50_Experiment_2_Fold_1_Setting_2'
project_dir = '/home/featurize/Results'
work_dir = '/home/featurize/Results/Size_50_Experiment_2_Fold_1_Setting_2'
save_path = '/home/featurize/Results/Size_50_Experiment_2_Fold_1_Setting_2'
dataset_type = 'GeospatialDataset'
data_root = '/home/featurize/Dataset_2_50_Experiment_2Fold_1'
img_norm_cfg = dict(
    means=[
        1122.3895263671875, 1111.793701171875, 1074.912353515625,
        1039.622802734375, 1037.2821044921875, 1084.11865234375,
        852.8363037109375, 872.4114379882812, 902.127685546875,
        939.39306640625, 952.8056640625, 1083.55859375, 1120.61279296875,
        1122.8453369140625, 1050.8677978515625, 999.6224975585938,
        966.1929931640625, 990.528076171875, 2056.760498046875,
        2219.82080078125, 2431.9755859375, 2703.125244140625, 2929.904296875,
        2912.407958984375, 2864.94384765625, 2788.756591796875,
        2571.685302734375, 2346.31591796875, 2196.347900390625,
        2134.359130859375, 1543.988525390625, 1719.965576171875,
        1853.1826171875, 1958.3668212890625, 2005.1046142578125,
        2237.401611328125
    ],
    stds=[
        88.0251693725586, 90.96491241455078, 227.46484375, 197.58360290527344,
        320.06024169921875, 513.6522216796875, 154.38479614257812,
        148.701416015625, 151.92620849609375, 171.6962890625, 172.4609375,
        142.61561584472656, 131.3424530029297, 135.05682373046875,
        250.5956573486328, 216.20228576660156, 352.9932556152344,
        570.7435913085938, 177.6968994140625, 157.25930786132812,
        192.72463989257812, 180.0116424560547, 169.9145965576172,
        168.2208251953125, 166.5236358642578, 173.964111328125,
        267.6305236816406, 232.35374450683594, 351.2685852050781,
        566.362060546875, 179.79954528808594, 165.3571319580078,
        165.59378051757812, 169.21348571777344, 184.45294189453125,
        162.64669799804688
    ])
bands = [0, 1, 2, 3, 4, 5]
tile_size = 366
crop_size = (366, 366)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            1122.3895263671875, 1111.793701171875, 1074.912353515625,
            1039.622802734375, 1037.2821044921875, 1084.11865234375,
            852.8363037109375, 872.4114379882812, 902.127685546875,
            939.39306640625, 952.8056640625, 1083.55859375, 1120.61279296875,
            1122.8453369140625, 1050.8677978515625, 999.6224975585938,
            966.1929931640625, 990.528076171875, 2056.760498046875,
            2219.82080078125, 2431.9755859375, 2703.125244140625,
            2929.904296875, 2912.407958984375, 2864.94384765625,
            2788.756591796875, 2571.685302734375, 2346.31591796875,
            2196.347900390625, 2134.359130859375, 1543.988525390625,
            1719.965576171875, 1853.1826171875, 1958.3668212890625,
            2005.1046142578125, 2237.401611328125
        ],
        stds=[
            88.0251693725586, 90.96491241455078, 227.46484375,
            197.58360290527344, 320.06024169921875, 513.6522216796875,
            154.38479614257812, 148.701416015625, 151.92620849609375,
            171.6962890625, 172.4609375, 142.61561584472656, 131.3424530029297,
            135.05682373046875, 250.5956573486328, 216.20228576660156,
            352.9932556152344, 570.7435913085938, 177.6968994140625,
            157.25930786132812, 192.72463989257812, 180.0116424560547,
            169.9145965576172, 168.2208251953125, 166.5236358642578,
            173.964111328125, 267.6305236816406, 232.35374450683594,
            351.2685852050781, 566.362060546875, 179.79954528808594,
            165.3571319580078, 165.59378051757812, 169.21348571777344,
            184.45294189453125, 162.64669799804688
        ]),
    dict(type='TorchRandomCrop', crop_size=(366, 366)),
    dict(type='Reshape', keys=['img'], new_shape=(6, 6, 366, 366)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, 366, 366)),
    dict(
        type='CastTensor',
        keys=['gt_semantic_seg'],
        new_type='torch.LongTensor'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True),
    dict(type='ToTensor', keys=['img']),
    dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
    dict(
        type='TorchNormalize',
        means=[
            1122.3895263671875, 1111.793701171875, 1074.912353515625,
            1039.622802734375, 1037.2821044921875, 1084.11865234375,
            852.8363037109375, 872.4114379882812, 902.127685546875,
            939.39306640625, 952.8056640625, 1083.55859375, 1120.61279296875,
            1122.8453369140625, 1050.8677978515625, 999.6224975585938,
            966.1929931640625, 990.528076171875, 2056.760498046875,
            2219.82080078125, 2431.9755859375, 2703.125244140625,
            2929.904296875, 2912.407958984375, 2864.94384765625,
            2788.756591796875, 2571.685302734375, 2346.31591796875,
            2196.347900390625, 2134.359130859375, 1543.988525390625,
            1719.965576171875, 1853.1826171875, 1958.3668212890625,
            2005.1046142578125, 2237.401611328125
        ],
        stds=[
            88.0251693725586, 90.96491241455078, 227.46484375,
            197.58360290527344, 320.06024169921875, 513.6522216796875,
            154.38479614257812, 148.701416015625, 151.92620849609375,
            171.6962890625, 172.4609375, 142.61561584472656, 131.3424530029297,
            135.05682373046875, 250.5956573486328, 216.20228576660156,
            352.9932556152344, 570.7435913085938, 177.6968994140625,
            157.25930786132812, 192.72463989257812, 180.0116424560547,
            169.9145965576172, 168.2208251953125, 166.5236358642578,
            173.964111328125, 267.6305236816406, 232.35374450683594,
            351.2685852050781, 566.362060546875, 179.79954528808594,
            165.3571319580078, 165.59378051757812, 169.21348571777344,
            184.45294189453125, 162.64669799804688
        ]),
    dict(
        type='Reshape',
        keys=['img'],
        new_shape=(6, 6, -1, -1),
        look_up=dict({
            '2': 1,
            '3': 2
        })),
    dict(type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
    dict(
        type='CollectTestList',
        keys=['img'],
        meta_keys=[
            'img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename',
            'ori_filename', 'img', 'img_shape', 'ori_shape', 'pad_shape',
            'scale_factor', 'img_norm_cfg'
        ])
]
CLASSES = ('Wheat', 'Maize', 'Sorghum', 'Barley', 'Rye', 'Oats', 'Grapes',
           'Rapeseed', 'Sunflower', 'Potatoes', 'Peas')
dataset = 'GeospatialDataset'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(
        type='GeospatialDataset',
        CLASSES=('Wheat', 'Maize', 'Sorghum', 'Barley', 'Rye', 'Oats',
                 'Grapes', 'Rapeseed', 'Sunflower', 'Potatoes', 'Peas'),
        reduce_zero_label=True,
        data_root='/home/featurize/Dataset_2_50_Experiment_2Fold_1',
        img_dir='Training_Set',
        ann_dir='Training_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1122.3895263671875, 1111.793701171875, 1074.912353515625,
                    1039.622802734375, 1037.2821044921875, 1084.11865234375,
                    852.8363037109375, 872.4114379882812, 902.127685546875,
                    939.39306640625, 952.8056640625, 1083.55859375,
                    1120.61279296875, 1122.8453369140625, 1050.8677978515625,
                    999.6224975585938, 966.1929931640625, 990.528076171875,
                    2056.760498046875, 2219.82080078125, 2431.9755859375,
                    2703.125244140625, 2929.904296875, 2912.407958984375,
                    2864.94384765625, 2788.756591796875, 2571.685302734375,
                    2346.31591796875, 2196.347900390625, 2134.359130859375,
                    1543.988525390625, 1719.965576171875, 1853.1826171875,
                    1958.3668212890625, 2005.1046142578125, 2237.401611328125
                ],
                stds=[
                    88.0251693725586, 90.96491241455078, 227.46484375,
                    197.58360290527344, 320.06024169921875, 513.6522216796875,
                    154.38479614257812, 148.701416015625, 151.92620849609375,
                    171.6962890625, 172.4609375, 142.61561584472656,
                    131.3424530029297, 135.05682373046875, 250.5956573486328,
                    216.20228576660156, 352.9932556152344, 570.7435913085938,
                    177.6968994140625, 157.25930786132812, 192.72463989257812,
                    180.0116424560547, 169.9145965576172, 168.2208251953125,
                    166.5236358642578, 173.964111328125, 267.6305236816406,
                    232.35374450683594, 351.2685852050781, 566.362060546875,
                    179.79954528808594, 165.3571319580078, 165.59378051757812,
                    169.21348571777344, 184.45294189453125, 162.64669799804688
                ]),
            dict(type='TorchRandomCrop', crop_size=(366, 366)),
            dict(type='Reshape', keys=['img'], new_shape=(6, 6, 366, 366)),
            dict(
                type='Reshape',
                keys=['gt_semantic_seg'],
                new_shape=(1, 366, 366)),
            dict(
                type='CastTensor',
                keys=['gt_semantic_seg'],
                new_type='torch.LongTensor'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        img_suffix='_image.npy',
        seg_map_suffix='_labels.npy'),
    val=dict(
        type='GeospatialDataset',
        CLASSES=('Wheat', 'Maize', 'Sorghum', 'Barley', 'Rye', 'Oats',
                 'Grapes', 'Rapeseed', 'Sunflower', 'Potatoes', 'Peas'),
        reduce_zero_label=True,
        data_root='/home/featurize/Dataset_2_50_Experiment_2Fold_1',
        img_dir='Validation_Set',
        ann_dir='Validation_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1122.3895263671875, 1111.793701171875, 1074.912353515625,
                    1039.622802734375, 1037.2821044921875, 1084.11865234375,
                    852.8363037109375, 872.4114379882812, 902.127685546875,
                    939.39306640625, 952.8056640625, 1083.55859375,
                    1120.61279296875, 1122.8453369140625, 1050.8677978515625,
                    999.6224975585938, 966.1929931640625, 990.528076171875,
                    2056.760498046875, 2219.82080078125, 2431.9755859375,
                    2703.125244140625, 2929.904296875, 2912.407958984375,
                    2864.94384765625, 2788.756591796875, 2571.685302734375,
                    2346.31591796875, 2196.347900390625, 2134.359130859375,
                    1543.988525390625, 1719.965576171875, 1853.1826171875,
                    1958.3668212890625, 2005.1046142578125, 2237.401611328125
                ],
                stds=[
                    88.0251693725586, 90.96491241455078, 227.46484375,
                    197.58360290527344, 320.06024169921875, 513.6522216796875,
                    154.38479614257812, 148.701416015625, 151.92620849609375,
                    171.6962890625, 172.4609375, 142.61561584472656,
                    131.3424530029297, 135.05682373046875, 250.5956573486328,
                    216.20228576660156, 352.9932556152344, 570.7435913085938,
                    177.6968994140625, 157.25930786132812, 192.72463989257812,
                    180.0116424560547, 169.9145965576172, 168.2208251953125,
                    166.5236358642578, 173.964111328125, 267.6305236816406,
                    232.35374450683594, 351.2685852050781, 566.362060546875,
                    179.79954528808594, 165.3571319580078, 165.59378051757812,
                    169.21348571777344, 184.45294189453125, 162.64669799804688
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 6, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        img_suffix='_image.npy',
        seg_map_suffix='_labels.npy'),
    test=dict(
        type='GeospatialDataset',
        CLASSES=('Wheat', 'Maize', 'Sorghum', 'Barley', 'Rye', 'Oats',
                 'Grapes', 'Rapeseed', 'Sunflower', 'Potatoes', 'Peas'),
        reduce_zero_label=True,
        data_root='/home/featurize/Dataset_2_50_Experiment_2Fold_1',
        img_dir='Test_Set',
        ann_dir='Test_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1122.3895263671875, 1111.793701171875, 1074.912353515625,
                    1039.622802734375, 1037.2821044921875, 1084.11865234375,
                    852.8363037109375, 872.4114379882812, 902.127685546875,
                    939.39306640625, 952.8056640625, 1083.55859375,
                    1120.61279296875, 1122.8453369140625, 1050.8677978515625,
                    999.6224975585938, 966.1929931640625, 990.528076171875,
                    2056.760498046875, 2219.82080078125, 2431.9755859375,
                    2703.125244140625, 2929.904296875, 2912.407958984375,
                    2864.94384765625, 2788.756591796875, 2571.685302734375,
                    2346.31591796875, 2196.347900390625, 2134.359130859375,
                    1543.988525390625, 1719.965576171875, 1853.1826171875,
                    1958.3668212890625, 2005.1046142578125, 2237.401611328125
                ],
                stds=[
                    88.0251693725586, 90.96491241455078, 227.46484375,
                    197.58360290527344, 320.06024169921875, 513.6522216796875,
                    154.38479614257812, 148.701416015625, 151.92620849609375,
                    171.6962890625, 172.4609375, 142.61561584472656,
                    131.3424530029297, 135.05682373046875, 250.5956573486328,
                    216.20228576660156, 352.9932556152344, 570.7435913085938,
                    177.6968994140625, 157.25930786132812, 192.72463989257812,
                    180.0116424560547, 169.9145965576172, 168.2208251953125,
                    166.5236358642578, 173.964111328125, 267.6305236816406,
                    232.35374450683594, 351.2685852050781, 566.362060546875,
                    179.79954528808594, 165.3571319580078, 165.59378051757812,
                    169.21348571777344, 184.45294189453125, 162.64669799804688
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(6, 6, -1, -1),
                look_up=dict({
                    '2': 1,
                    '3': 2
                })),
            dict(
                type='CastTensor', keys=['img'], new_type='torch.FloatTensor'),
            dict(
                type='CollectTestList',
                keys=['img'],
                meta_keys=[
                    'img_info', 'seg_fields', 'img_prefix', 'seg_prefix',
                    'filename', 'ori_filename', 'img', 'img_shape',
                    'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg'
                ])
        ],
        img_suffix='_image.npy',
        seg_map_suffix='_labels.npy'))
optimizer = dict(
    type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(
    by_epoch=True,
    interval=100,
    out_dir='/home/featurize/Results/Size_50_Experiment_2_Fold_1_Setting_2')
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mDice', 'mFscore'],
    pre_eval=True,
    save_best='mIoU',
    by_epoch=True)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type='EpochBasedRunner', max_epochs=20)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained=
        '/home/featurize/work/hls-foundation-os/Current_Pretrained_Prithvi_Weights/Prithvi_EO_V1_100M.pt',
        img_size=366,
        patch_size=61,
        num_frames=6,
        tubelet_size=1,
        in_chans=6,
        embed_dim=768,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=4608,
        output_embed_dim=4608,
        drop_cls_token=True,
        Hp=6,
        Wp=6),
    decode_head=dict(
        num_classes=11,
        in_channels=4608,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[
                48.24197892603771, 1.3908071469704752, 13.91045298879203,
                0.16260287710088167, 24.915739196406832, 1.04194004835998,
                0.44759908979569946, 2.6068422990207574, 13.745187372306951,
                33.410956287558186, 3.143579631887296
            ],
            avg_non_ignore=True)),
    auxiliary_head=dict(
        num_classes=11,
        in_channels=4608,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[
                48.24197892603771, 1.3908071469704752, 13.91045298879203,
                0.16260287710088167, 24.915739196406832, 1.04194004835998,
                0.44759908979569946, 2.6068422990207574, 13.745187372306951,
                33.410956287558186, 3.143579631887296
            ],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(183, 183), crop_size=(366, 366)))
auto_resume = False
gpu_ids = range(0, 1)
