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
    0.17943676555320973, 1.3472306749861818, 7.788549609514691,
    0.5933110303234815, 33.44558379154902, 2.632246645198013,
    49.45546440082039, 0.8937316499127668, 1.3150492291810718,
    2.8111886947216775, 4.85870754173863
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=[
        0.17943676555320973, 1.3472306749861818, 7.788549609514691,
        0.5933110303234815, 33.44558379154902, 2.632246645198013,
        49.45546440082039, 0.8937316499127668, 1.3150492291810718,
        2.8111886947216775, 4.85870754173863
    ],
    avg_non_ignore=True)
output_embed_dim = 4608
experiment = 'Size_5_Experiment_3_Fold_1_Setting_1'
project_dir = '/home/featurize/Results'
work_dir = '/home/featurize/Results/Size_5_Experiment_3_Fold_1_Setting_1'
save_path = '/home/featurize/Results/Size_5_Experiment_3_Fold_1_Setting_1'
dataset_type = 'GeospatialDataset'
data_root = '/home/featurize/Dataset_1_5_Experiment_3Fold_1'
img_norm_cfg = dict(
    means=[
        1087.8736572265625, 996.471923828125, 965.0195922851562,
        984.8784790039062, 956.47607421875, 1017.7380981445312,
        1073.03466796875, 1140.4896240234375, 1054.77978515625,
        1020.0550537109375, 981.1219482421875, 905.3924560546875,
        994.7728271484375, 917.841552734375, 833.7246704101562,
        810.3518676757812, 729.974365234375, 874.8143920898438, 1065.396484375,
        1123.212890625, 1057.910400390625, 918.9129638671875,
        816.3870239257812, 686.6246337890625
    ],
    stds=[
        392.0556640625, 161.1523895263672, 94.90672302246094,
        118.73157501220703, 231.76856994628906, 86.99456787109375,
        58.04689407348633, 178.5121612548828, 61.517452239990234,
        310.8953857421875, 345.74139404296875, 467.0456848144531,
        472.1729431152344, 191.4079132080078, 134.80113220214844,
        161.3466033935547, 275.5858459472656, 100.39026641845703,
        116.68968963623047, 190.22354125976562, 100.7003402709961,
        349.4181213378906, 396.1525573730469, 539.5966186523438
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
            1087.8736572265625, 996.471923828125, 965.0195922851562,
            984.8784790039062, 956.47607421875, 1017.7380981445312,
            1073.03466796875, 1140.4896240234375, 1054.77978515625,
            1020.0550537109375, 981.1219482421875, 905.3924560546875,
            994.7728271484375, 917.841552734375, 833.7246704101562,
            810.3518676757812, 729.974365234375, 874.8143920898438,
            1065.396484375, 1123.212890625, 1057.910400390625,
            918.9129638671875, 816.3870239257812, 686.6246337890625
        ],
        stds=[
            392.0556640625, 161.1523895263672, 94.90672302246094,
            118.73157501220703, 231.76856994628906, 86.99456787109375,
            58.04689407348633, 178.5121612548828, 61.517452239990234,
            310.8953857421875, 345.74139404296875, 467.0456848144531,
            472.1729431152344, 191.4079132080078, 134.80113220214844,
            161.3466033935547, 275.5858459472656, 100.39026641845703,
            116.68968963623047, 190.22354125976562, 100.7003402709961,
            349.4181213378906, 396.1525573730469, 539.5966186523438
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
            1087.8736572265625, 996.471923828125, 965.0195922851562,
            984.8784790039062, 956.47607421875, 1017.7380981445312,
            1073.03466796875, 1140.4896240234375, 1054.77978515625,
            1020.0550537109375, 981.1219482421875, 905.3924560546875,
            994.7728271484375, 917.841552734375, 833.7246704101562,
            810.3518676757812, 729.974365234375, 874.8143920898438,
            1065.396484375, 1123.212890625, 1057.910400390625,
            918.9129638671875, 816.3870239257812, 686.6246337890625
        ],
        stds=[
            392.0556640625, 161.1523895263672, 94.90672302246094,
            118.73157501220703, 231.76856994628906, 86.99456787109375,
            58.04689407348633, 178.5121612548828, 61.517452239990234,
            310.8953857421875, 345.74139404296875, 467.0456848144531,
            472.1729431152344, 191.4079132080078, 134.80113220214844,
            161.3466033935547, 275.5858459472656, 100.39026641845703,
            116.68968963623047, 190.22354125976562, 100.7003402709961,
            349.4181213378906, 396.1525573730469, 539.5966186523438
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
        data_root='/home/featurize/Dataset_1_5_Experiment_3Fold_1',
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
                    1087.8736572265625, 996.471923828125, 965.0195922851562,
                    984.8784790039062, 956.47607421875, 1017.7380981445312,
                    1073.03466796875, 1140.4896240234375, 1054.77978515625,
                    1020.0550537109375, 981.1219482421875, 905.3924560546875,
                    994.7728271484375, 917.841552734375, 833.7246704101562,
                    810.3518676757812, 729.974365234375, 874.8143920898438,
                    1065.396484375, 1123.212890625, 1057.910400390625,
                    918.9129638671875, 816.3870239257812, 686.6246337890625
                ],
                stds=[
                    392.0556640625, 161.1523895263672, 94.90672302246094,
                    118.73157501220703, 231.76856994628906, 86.99456787109375,
                    58.04689407348633, 178.5121612548828, 61.517452239990234,
                    310.8953857421875, 345.74139404296875, 467.0456848144531,
                    472.1729431152344, 191.4079132080078, 134.80113220214844,
                    161.3466033935547, 275.5858459472656, 100.39026641845703,
                    116.68968963623047, 190.22354125976562, 100.7003402709961,
                    349.4181213378906, 396.1525573730469, 539.5966186523438
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
        data_root='/home/featurize/Dataset_1_5_Experiment_3Fold_1',
        img_dir='Validation_Set',
        ann_dir='Validation_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1087.8736572265625, 996.471923828125, 965.0195922851562,
                    984.8784790039062, 956.47607421875, 1017.7380981445312,
                    1073.03466796875, 1140.4896240234375, 1054.77978515625,
                    1020.0550537109375, 981.1219482421875, 905.3924560546875,
                    994.7728271484375, 917.841552734375, 833.7246704101562,
                    810.3518676757812, 729.974365234375, 874.8143920898438,
                    1065.396484375, 1123.212890625, 1057.910400390625,
                    918.9129638671875, 816.3870239257812, 686.6246337890625
                ],
                stds=[
                    392.0556640625, 161.1523895263672, 94.90672302246094,
                    118.73157501220703, 231.76856994628906, 86.99456787109375,
                    58.04689407348633, 178.5121612548828, 61.517452239990234,
                    310.8953857421875, 345.74139404296875, 467.0456848144531,
                    472.1729431152344, 191.4079132080078, 134.80113220214844,
                    161.3466033935547, 275.5858459472656, 100.39026641845703,
                    116.68968963623047, 190.22354125976562, 100.7003402709961,
                    349.4181213378906, 396.1525573730469, 539.5966186523438
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
        data_root='/home/featurize/Dataset_1_5_Experiment_3Fold_1',
        img_dir='Test_Set',
        ann_dir='Test_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1087.8736572265625, 996.471923828125, 965.0195922851562,
                    984.8784790039062, 956.47607421875, 1017.7380981445312,
                    1073.03466796875, 1140.4896240234375, 1054.77978515625,
                    1020.0550537109375, 981.1219482421875, 905.3924560546875,
                    994.7728271484375, 917.841552734375, 833.7246704101562,
                    810.3518676757812, 729.974365234375, 874.8143920898438,
                    1065.396484375, 1123.212890625, 1057.910400390625,
                    918.9129638671875, 816.3870239257812, 686.6246337890625
                ],
                stds=[
                    392.0556640625, 161.1523895263672, 94.90672302246094,
                    118.73157501220703, 231.76856994628906, 86.99456787109375,
                    58.04689407348633, 178.5121612548828, 61.517452239990234,
                    310.8953857421875, 345.74139404296875, 467.0456848144531,
                    472.1729431152344, 191.4079132080078, 134.80113220214844,
                    161.3466033935547, 275.5858459472656, 100.39026641845703,
                    116.68968963623047, 190.22354125976562, 100.7003402709961,
                    349.4181213378906, 396.1525573730469, 539.5966186523438
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
    out_dir='/home/featurize/Results/Size_5_Experiment_3_Fold_1_Setting_1')
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
                0.17943676555320973, 1.3472306749861818, 7.788549609514691,
                0.5933110303234815, 33.44558379154902, 2.632246645198013,
                49.45546440082039, 0.8937316499127668, 1.3150492291810718,
                2.8111886947216775, 4.85870754173863
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
                0.17943676555320973, 1.3472306749861818, 7.788549609514691,
                0.5933110303234815, 33.44558379154902, 2.632246645198013,
                49.45546440082039, 0.8937316499127668, 1.3150492291810718,
                2.8111886947216775, 4.85870754173863
            ],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(183, 183), crop_size=(366, 366)))
auto_resume = False
gpu_ids = range(0, 1)
