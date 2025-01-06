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
    0.1805498368493623, 1.2891398917579358, 8.215700647347298,
    0.660529398896495, 39.767097445728595, 2.2317346067409605,
    38.797124286401385, 0.9048376427700657, 1.2196343941121224,
    2.509129847797349, 4.398306944951367
]
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=[
        0.1805498368493623, 1.2891398917579358, 8.215700647347298,
        0.660529398896495, 39.767097445728595, 2.2317346067409605,
        38.797124286401385, 0.9048376427700657, 1.2196343941121224,
        2.509129847797349, 4.398306944951367
    ],
    avg_non_ignore=True)
output_embed_dim = 4608
experiment = 'Size_25_Experiment_3_Fold_1_Setting_2'
project_dir = '/home/featurize/Results'
work_dir = '/home/featurize/Results/Size_25_Experiment_3_Fold_1_Setting_2'
save_path = '/home/featurize/Results/Size_25_Experiment_3_Fold_1_Setting_2'
dataset_type = 'GeospatialDataset'
data_root = '/home/featurize/Dataset_2_25_Experiment_3Fold_1'
img_norm_cfg = dict(
    means=[
        1074.1385498046875, 1119.3740234375, 1054.8212890625,
        1050.5323486328125, 1010.7451171875, 943.6336669921875,
        923.8756713867188, 936.591796875, 835.8751220703125, 807.0238647460938,
        733.2495727539062, 863.6655883789062, 1065.8094482421875,
        1094.4266357421875, 1053.9906005859375, 947.4544067382812,
        842.326416015625, 723.4024658203125, 2146.501708984375,
        2513.42236328125, 3022.88671875, 3388.59033203125, 3841.516845703125,
        3758.624755859375, 3609.43359375, 3222.963623046875, 2956.281005859375,
        2746.011474609375, 2543.30078125, 2304.1123046875, 1670.60546875,
        2004.779052734375, 2056.456298828125, 1992.809326171875,
        1859.998779296875, 2069.95458984375
    ],
    stds=[
        60.199222564697266, 188.88772583007812, 62.79337692260742,
        357.0793151855469, 332.0881652832031, 451.9391174316406,
        306.04443359375, 223.84036254882812, 131.6476287841797,
        161.0316619873047, 266.571044921875, 82.27179718017578,
        122.2155532836914, 199.59107971191406, 103.23735809326172,
        401.8136901855469, 383.4722595214844, 523.5040283203125,
        282.3654479980469, 169.770751953125, 146.03768920898438,
        150.8448944091797, 202.09844970703125, 139.3668670654297,
        118.4309310913086, 190.49740600585938, 146.68023681640625,
        398.2655029296875, 382.3602294921875, 585.155517578125,
        353.71954345703125, 139.32225036621094, 118.53516387939453,
        153.41006469726562, 273.8182678222656, 99.3881607055664
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
            1074.1385498046875, 1119.3740234375, 1054.8212890625,
            1050.5323486328125, 1010.7451171875, 943.6336669921875,
            923.8756713867188, 936.591796875, 835.8751220703125,
            807.0238647460938, 733.2495727539062, 863.6655883789062,
            1065.8094482421875, 1094.4266357421875, 1053.9906005859375,
            947.4544067382812, 842.326416015625, 723.4024658203125,
            2146.501708984375, 2513.42236328125, 3022.88671875,
            3388.59033203125, 3841.516845703125, 3758.624755859375,
            3609.43359375, 3222.963623046875, 2956.281005859375,
            2746.011474609375, 2543.30078125, 2304.1123046875, 1670.60546875,
            2004.779052734375, 2056.456298828125, 1992.809326171875,
            1859.998779296875, 2069.95458984375
        ],
        stds=[
            60.199222564697266, 188.88772583007812, 62.79337692260742,
            357.0793151855469, 332.0881652832031, 451.9391174316406,
            306.04443359375, 223.84036254882812, 131.6476287841797,
            161.0316619873047, 266.571044921875, 82.27179718017578,
            122.2155532836914, 199.59107971191406, 103.23735809326172,
            401.8136901855469, 383.4722595214844, 523.5040283203125,
            282.3654479980469, 169.770751953125, 146.03768920898438,
            150.8448944091797, 202.09844970703125, 139.3668670654297,
            118.4309310913086, 190.49740600585938, 146.68023681640625,
            398.2655029296875, 382.3602294921875, 585.155517578125,
            353.71954345703125, 139.32225036621094, 118.53516387939453,
            153.41006469726562, 273.8182678222656, 99.3881607055664
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
            1074.1385498046875, 1119.3740234375, 1054.8212890625,
            1050.5323486328125, 1010.7451171875, 943.6336669921875,
            923.8756713867188, 936.591796875, 835.8751220703125,
            807.0238647460938, 733.2495727539062, 863.6655883789062,
            1065.8094482421875, 1094.4266357421875, 1053.9906005859375,
            947.4544067382812, 842.326416015625, 723.4024658203125,
            2146.501708984375, 2513.42236328125, 3022.88671875,
            3388.59033203125, 3841.516845703125, 3758.624755859375,
            3609.43359375, 3222.963623046875, 2956.281005859375,
            2746.011474609375, 2543.30078125, 2304.1123046875, 1670.60546875,
            2004.779052734375, 2056.456298828125, 1992.809326171875,
            1859.998779296875, 2069.95458984375
        ],
        stds=[
            60.199222564697266, 188.88772583007812, 62.79337692260742,
            357.0793151855469, 332.0881652832031, 451.9391174316406,
            306.04443359375, 223.84036254882812, 131.6476287841797,
            161.0316619873047, 266.571044921875, 82.27179718017578,
            122.2155532836914, 199.59107971191406, 103.23735809326172,
            401.8136901855469, 383.4722595214844, 523.5040283203125,
            282.3654479980469, 169.770751953125, 146.03768920898438,
            150.8448944091797, 202.09844970703125, 139.3668670654297,
            118.4309310913086, 190.49740600585938, 146.68023681640625,
            398.2655029296875, 382.3602294921875, 585.155517578125,
            353.71954345703125, 139.32225036621094, 118.53516387939453,
            153.41006469726562, 273.8182678222656, 99.3881607055664
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
        data_root='/home/featurize/Dataset_2_25_Experiment_3Fold_1',
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
                    1074.1385498046875, 1119.3740234375, 1054.8212890625,
                    1050.5323486328125, 1010.7451171875, 943.6336669921875,
                    923.8756713867188, 936.591796875, 835.8751220703125,
                    807.0238647460938, 733.2495727539062, 863.6655883789062,
                    1065.8094482421875, 1094.4266357421875, 1053.9906005859375,
                    947.4544067382812, 842.326416015625, 723.4024658203125,
                    2146.501708984375, 2513.42236328125, 3022.88671875,
                    3388.59033203125, 3841.516845703125, 3758.624755859375,
                    3609.43359375, 3222.963623046875, 2956.281005859375,
                    2746.011474609375, 2543.30078125, 2304.1123046875,
                    1670.60546875, 2004.779052734375, 2056.456298828125,
                    1992.809326171875, 1859.998779296875, 2069.95458984375
                ],
                stds=[
                    60.199222564697266, 188.88772583007812, 62.79337692260742,
                    357.0793151855469, 332.0881652832031, 451.9391174316406,
                    306.04443359375, 223.84036254882812, 131.6476287841797,
                    161.0316619873047, 266.571044921875, 82.27179718017578,
                    122.2155532836914, 199.59107971191406, 103.23735809326172,
                    401.8136901855469, 383.4722595214844, 523.5040283203125,
                    282.3654479980469, 169.770751953125, 146.03768920898438,
                    150.8448944091797, 202.09844970703125, 139.3668670654297,
                    118.4309310913086, 190.49740600585938, 146.68023681640625,
                    398.2655029296875, 382.3602294921875, 585.155517578125,
                    353.71954345703125, 139.32225036621094, 118.53516387939453,
                    153.41006469726562, 273.8182678222656, 99.3881607055664
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
        data_root='/home/featurize/Dataset_2_25_Experiment_3Fold_1',
        img_dir='Validation_Set',
        ann_dir='Validation_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1074.1385498046875, 1119.3740234375, 1054.8212890625,
                    1050.5323486328125, 1010.7451171875, 943.6336669921875,
                    923.8756713867188, 936.591796875, 835.8751220703125,
                    807.0238647460938, 733.2495727539062, 863.6655883789062,
                    1065.8094482421875, 1094.4266357421875, 1053.9906005859375,
                    947.4544067382812, 842.326416015625, 723.4024658203125,
                    2146.501708984375, 2513.42236328125, 3022.88671875,
                    3388.59033203125, 3841.516845703125, 3758.624755859375,
                    3609.43359375, 3222.963623046875, 2956.281005859375,
                    2746.011474609375, 2543.30078125, 2304.1123046875,
                    1670.60546875, 2004.779052734375, 2056.456298828125,
                    1992.809326171875, 1859.998779296875, 2069.95458984375
                ],
                stds=[
                    60.199222564697266, 188.88772583007812, 62.79337692260742,
                    357.0793151855469, 332.0881652832031, 451.9391174316406,
                    306.04443359375, 223.84036254882812, 131.6476287841797,
                    161.0316619873047, 266.571044921875, 82.27179718017578,
                    122.2155532836914, 199.59107971191406, 103.23735809326172,
                    401.8136901855469, 383.4722595214844, 523.5040283203125,
                    282.3654479980469, 169.770751953125, 146.03768920898438,
                    150.8448944091797, 202.09844970703125, 139.3668670654297,
                    118.4309310913086, 190.49740600585938, 146.68023681640625,
                    398.2655029296875, 382.3602294921875, 585.155517578125,
                    353.71954345703125, 139.32225036621094, 118.53516387939453,
                    153.41006469726562, 273.8182678222656, 99.3881607055664
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
        data_root='/home/featurize/Dataset_2_25_Experiment_3Fold_1',
        img_dir='Test_Set',
        ann_dir='Test_Set',
        pipeline=[
            dict(type='LoadGeospatialImageFromFile', to_float32=True),
            dict(type='ToTensor', keys=['img']),
            dict(type='TorchPermute', keys=['img'], order=(2, 0, 1)),
            dict(
                type='TorchNormalize',
                means=[
                    1074.1385498046875, 1119.3740234375, 1054.8212890625,
                    1050.5323486328125, 1010.7451171875, 943.6336669921875,
                    923.8756713867188, 936.591796875, 835.8751220703125,
                    807.0238647460938, 733.2495727539062, 863.6655883789062,
                    1065.8094482421875, 1094.4266357421875, 1053.9906005859375,
                    947.4544067382812, 842.326416015625, 723.4024658203125,
                    2146.501708984375, 2513.42236328125, 3022.88671875,
                    3388.59033203125, 3841.516845703125, 3758.624755859375,
                    3609.43359375, 3222.963623046875, 2956.281005859375,
                    2746.011474609375, 2543.30078125, 2304.1123046875,
                    1670.60546875, 2004.779052734375, 2056.456298828125,
                    1992.809326171875, 1859.998779296875, 2069.95458984375
                ],
                stds=[
                    60.199222564697266, 188.88772583007812, 62.79337692260742,
                    357.0793151855469, 332.0881652832031, 451.9391174316406,
                    306.04443359375, 223.84036254882812, 131.6476287841797,
                    161.0316619873047, 266.571044921875, 82.27179718017578,
                    122.2155532836914, 199.59107971191406, 103.23735809326172,
                    401.8136901855469, 383.4722595214844, 523.5040283203125,
                    282.3654479980469, 169.770751953125, 146.03768920898438,
                    150.8448944091797, 202.09844970703125, 139.3668670654297,
                    118.4309310913086, 190.49740600585938, 146.68023681640625,
                    398.2655029296875, 382.3602294921875, 585.155517578125,
                    353.71954345703125, 139.32225036621094, 118.53516387939453,
                    153.41006469726562, 273.8182678222656, 99.3881607055664
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
    out_dir='/home/featurize/Results/Size_25_Experiment_3_Fold_1_Setting_2')
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
                0.1805498368493623, 1.2891398917579358, 8.215700647347298,
                0.660529398896495, 39.767097445728595, 2.2317346067409605,
                38.797124286401385, 0.9048376427700657, 1.2196343941121224,
                2.509129847797349, 4.398306944951367
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
                0.1805498368493623, 1.2891398917579358, 8.215700647347298,
                0.660529398896495, 39.767097445728595, 2.2317346067409605,
                38.797124286401385, 0.9048376427700657, 1.2196343941121224,
                2.509129847797349, 4.398306944951367
            ],
            avg_non_ignore=True)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(183, 183), crop_size=(366, 366)))
auto_resume = False
gpu_ids = range(0, 1)
