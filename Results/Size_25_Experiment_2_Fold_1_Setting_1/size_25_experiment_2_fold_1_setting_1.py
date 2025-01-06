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
    52.48395658357144, 0.9957708596012649, 15.839481257484874,
    0.16052980414389506, 23.582234989298186, 1.0191788795109697,
    0.5676662325606621, 2.3892025838362456, 9.962004430587463,
    43.04785659331114, 2.802322741997875
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=[
        52.48395658357144, 0.9957708596012649, 15.839481257484874,
        0.16052980414389506, 23.582234989298186, 1.0191788795109697,
        0.5676662325606621, 2.3892025838362456, 9.962004430587463,
        43.04785659331114, 2.802322741997875
    ],
    avg_non_ignore=True)
output_embed_dim = 4608
experiment = 'Size_25_Experiment_2_Fold_1_Setting_1'
project_dir = '/home/featurize/Results'
work_dir = '/home/featurize/Results/Size_25_Experiment_2_Fold_1_Setting_1'
save_path = '/home/featurize/Results/Size_25_Experiment_2_Fold_1_Setting_1'
dataset_type = 'GeospatialDataset'
data_root = '/home/featurize/Dataset_1_25_Experiment_2Fold_1'
img_norm_cfg = dict(
    means=[
        938.7228393554688, 935.4931640625, 943.4722900390625,
        998.3135986328125, 1035.7413330078125, 1102.42626953125,
        1102.3197021484375, 1090.181396484375, 1047.38330078125,
        1024.561279296875, 1008.697021484375, 1078.959716796875, 833.41015625,
        854.85595703125, 881.6832275390625, 918.17578125, 930.3850708007812,
        1059.1834716796875, 1088.4139404296875, 1089.2845458984375,
        1014.2647094726562, 978.3624267578125, 929.9804077148438,
        981.2182006835938
    ],
    stds=[
        131.77464294433594, 95.71418762207031, 116.06571197509766,
        128.84205627441406, 131.74510192871094, 112.97071838378906,
        84.1624755859375, 88.5306625366211, 193.71629333496094,
        211.2794952392578, 279.823486328125, 447.7043762207031,
        150.3252716064453, 120.39602661132812, 145.2886199951172,
        161.4945526123047, 171.84434509277344, 142.8815155029297,
        132.63180541992188, 136.47445678710938, 216.19776916503906,
        234.49913024902344, 310.82769775390625, 499.63775634765625
    ])
bands = [0, 1, 2, 3]
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
            938.7228393554688, 935.4931640625, 943.4722900390625,
            998.3135986328125, 1035.7413330078125, 1102.42626953125,
            1102.3197021484375, 1090.181396484375, 1047.38330078125,
            1024.561279296875, 1008.697021484375, 1078.959716796875,
            833.41015625, 854.85595703125, 881.6832275390625, 918.17578125,
            930.3850708007812, 1059.1834716796875, 1088.4139404296875,
            1089.2845458984375, 1014.2647094726562, 978.3624267578125,
            929.9804077148438, 981.2182006835938
        ],
        stds=[
            131.77464294433594, 95.71418762207031, 116.06571197509766,
            128.84205627441406, 131.74510192871094, 112.97071838378906,
            84.1624755859375, 88.5306625366211, 193.71629333496094,
            211.2794952392578, 279.823486328125, 447.7043762207031,
            150.3252716064453, 120.39602661132812, 145.2886199951172,
            161.4945526123047, 171.84434509277344, 142.8815155029297,
            132.63180541992188, 136.47445678710938, 216.19776916503906,
            234.49913024902344, 310.82769775390625, 499.63775634765625
        ]),
    dict(type='TorchRandomCrop', crop_size=(366, 366)),
    dict(type='Reshape', keys=['img'], new_shape=(4, 6, 366, 366)),
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
            938.7228393554688, 935.4931640625, 943.4722900390625,
            998.3135986328125, 1035.7413330078125, 1102.42626953125,
            1102.3197021484375, 1090.181396484375, 1047.38330078125,
            1024.561279296875, 1008.697021484375, 1078.959716796875,
            833.41015625, 854.85595703125, 881.6832275390625, 918.17578125,
            930.3850708007812, 1059.1834716796875, 1088.4139404296875,
            1089.2845458984375, 1014.2647094726562, 978.3624267578125,
            929.9804077148438, 981.2182006835938
        ],
        stds=[
            131.77464294433594, 95.71418762207031, 116.06571197509766,
            128.84205627441406, 131.74510192871094, 112.97071838378906,
            84.1624755859375, 88.5306625366211, 193.71629333496094,
            211.2794952392578, 279.823486328125, 447.7043762207031,
            150.3252716064453, 120.39602661132812, 145.2886199951172,
            161.4945526123047, 171.84434509277344, 142.8815155029297,
            132.63180541992188, 136.47445678710938, 216.19776916503906,
            234.49913024902344, 310.82769775390625, 499.63775634765625
        ]),
    dict(
        type='Reshape',
        keys=['img'],
        new_shape=(4, 6, -1, -1),
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
        data_root='/home/featurize/Dataset_1_25_Experiment_2Fold_1',
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
                    938.7228393554688, 935.4931640625, 943.4722900390625,
                    998.3135986328125, 1035.7413330078125, 1102.42626953125,
                    1102.3197021484375, 1090.181396484375, 1047.38330078125,
                    1024.561279296875, 1008.697021484375, 1078.959716796875,
                    833.41015625, 854.85595703125, 881.6832275390625,
                    918.17578125, 930.3850708007812, 1059.1834716796875,
                    1088.4139404296875, 1089.2845458984375, 1014.2647094726562,
                    978.3624267578125, 929.9804077148438, 981.2182006835938
                ],
                stds=[
                    131.77464294433594, 95.71418762207031, 116.06571197509766,
                    128.84205627441406, 131.74510192871094, 112.97071838378906,
                    84.1624755859375, 88.5306625366211, 193.71629333496094,
                    211.2794952392578, 279.823486328125, 447.7043762207031,
                    150.3252716064453, 120.39602661132812, 145.2886199951172,
                    161.4945526123047, 171.84434509277344, 142.8815155029297,
                    132.63180541992188, 136.47445678710938, 216.19776916503906,
                    234.49913024902344, 310.82769775390625, 499.63775634765625
                ]),
            dict(type='TorchRandomCrop', crop_size=(366, 366)),
            dict(type='Reshape', keys=['img'], new_shape=(4, 6, 366, 366)),
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
        data_root='/home/featurize/Dataset_1_25_Experiment_2Fold_1',
        img_dir='Validation_Set',
        ann_dir='Validation_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    938.7228393554688, 935.4931640625, 943.4722900390625,
                    998.3135986328125, 1035.7413330078125, 1102.42626953125,
                    1102.3197021484375, 1090.181396484375, 1047.38330078125,
                    1024.561279296875, 1008.697021484375, 1078.959716796875,
                    833.41015625, 854.85595703125, 881.6832275390625,
                    918.17578125, 930.3850708007812, 1059.1834716796875,
                    1088.4139404296875, 1089.2845458984375, 1014.2647094726562,
                    978.3624267578125, 929.9804077148438, 981.2182006835938
                ],
                stds=[
                    131.77464294433594, 95.71418762207031, 116.06571197509766,
                    128.84205627441406, 131.74510192871094, 112.97071838378906,
                    84.1624755859375, 88.5306625366211, 193.71629333496094,
                    211.2794952392578, 279.823486328125, 447.7043762207031,
                    150.3252716064453, 120.39602661132812, 145.2886199951172,
                    161.4945526123047, 171.84434509277344, 142.8815155029297,
                    132.63180541992188, 136.47445678710938, 216.19776916503906,
                    234.49913024902344, 310.82769775390625, 499.63775634765625
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(4, 6, -1, -1),
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
        data_root='/home/featurize/Dataset_1_25_Experiment_2Fold_1',
        img_dir='Test_Set',
        ann_dir='Test_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    938.7228393554688, 935.4931640625, 943.4722900390625,
                    998.3135986328125, 1035.7413330078125, 1102.42626953125,
                    1102.3197021484375, 1090.181396484375, 1047.38330078125,
                    1024.561279296875, 1008.697021484375, 1078.959716796875,
                    833.41015625, 854.85595703125, 881.6832275390625,
                    918.17578125, 930.3850708007812, 1059.1834716796875,
                    1088.4139404296875, 1089.2845458984375, 1014.2647094726562,
                    978.3624267578125, 929.9804077148438, 981.2182006835938
                ],
                stds=[
                    131.77464294433594, 95.71418762207031, 116.06571197509766,
                    128.84205627441406, 131.74510192871094, 112.97071838378906,
                    84.1624755859375, 88.5306625366211, 193.71629333496094,
                    211.2794952392578, 279.823486328125, 447.7043762207031,
                    150.3252716064453, 120.39602661132812, 145.2886199951172,
                    161.4945526123047, 171.84434509277344, 142.8815155029297,
                    132.63180541992188, 136.47445678710938, 216.19776916503906,
                    234.49913024902344, 310.82769775390625, 499.63775634765625
                ]),
            dict(
                type='Reshape',
                keys=['img'],
                new_shape=(4, 6, -1, -1),
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
    out_dir='/home/featurize/Results/Size_25_Experiment_2_Fold_1_Setting_1')
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
        in_chans=4,
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
                52.48395658357144, 0.9957708596012649, 15.839481257484874,
                0.16052980414389506, 23.582234989298186, 1.0191788795109697,
                0.5676662325606621, 2.3892025838362456, 9.962004430587463,
                43.04785659331114, 2.802322741997875
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
                52.48395658357144, 0.9957708596012649, 15.839481257484874,
                0.16052980414389506, 23.582234989298186, 1.0191788795109697,
                0.5676662325606621, 2.3892025838362456, 9.962004430587463,
                43.04785659331114, 2.802322741997875
            ],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(183, 183), crop_size=(366, 366)))
auto_resume = False
gpu_ids = range(0, 1)
