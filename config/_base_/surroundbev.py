# ================== data ========================
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
occ_path = "data/surroundocc/samples"
input_shape = (704, 256)
batch_size = 1

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadBEVLabelSurroundOcc", occ_path=occ_path, semantic=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadBEVLabelSurroundOcc", occ_path=occ_path, semantic=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

common_keys = [
    'img',
    'projection_mat',
    'image_wh',
    'bev_label',
    'occ_xyz',
    'bev_xyz',
    'bev_cam_mask',
    'ori_img',
    'cam_positions',
    'focal_positions'
]

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    return_keys=common_keys,
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_sweeps_occ.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    return_keys=common_keys,
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=2
)
