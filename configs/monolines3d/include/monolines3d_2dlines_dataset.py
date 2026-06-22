dataset_name = 'monolines3d'

data_aug_scales = [(640, 640)]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]

data_aug_scale_overlap = None
batch_size_train = 1
batch_size_val = 1

mono3d_dataset_root = 'data/monolines3d'
mono3d_train2d = True
mono3d_use_image_normalized_target_line_coords = False
mono3d_normalize_line_space = False
mono3d_preload_images = False
mono3d_do_not_normalize_images = False
mono3d_strict = False

use_lmap = False
