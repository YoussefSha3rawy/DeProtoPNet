base_architecture = 'densenet121'
img_size = 224

import datetime
experiment_run = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

import socket
hostname = socket.gethostname()
if hostname.endswith('local'):  # Example check for local machine names
    print("Running on Macbook locally")
    data_path = '/Users/youssefshaarawy/Documents/Datasets/OCT2017/'
    # data_path = '/Users/youssefshaarawy/Documents/Datasets/JustRAIGS'
    # data_path = '/Users/youssefshaarawy/Downloads/CUB_200_2011/CUB_200_2011/'
else:
    print(f"Running on remote server: {hostname}")
    data_path = "/users/adfx751/Datasets/OCT2017/"
    # data_path = '/users/adfx751/Datasets/JustRAIGS'
    # data_path = '/users/adfx751/Datasets/CUB_200_2011/'

# Full set: './datasets/CUB_200_2011/'
# Cropped set: './datasets/cub200_cropped/'
# Stanford dogs: './datasets/stanford_dogs/'
# data_path = './datasets/CUB_200_2011/'


#120 classes in stanford_dogs, 200 in CUB_200_2011
if 'stanford_dogs' in data_path:
    num_classes = 120
elif 'OCT2017' in data_path:
    num_classes = 4
elif 'JustRAIGS' in data_path:
    num_classes = 2
else:
    num_classes = 200

num_prototypes = num_classes * 10

# Cropped set: train_cropped & test_cropped
# Full set: train & test
train_dir = data_path + 'train_balanced'
test_dir = data_path + 'val/'
train_push_dir = data_path + 'train_balanced/'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3,
                       'conv_offset': 1e-4,
                       'joint_last_layer_lr': 1e-5}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

warm_pre_offset_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3,
                      'features': 1e-4}

warm_pre_prototype_optimizer_lrs = {'add_on_layers': 3e-3,
                      'conv_offset': 3e-3,
                      'features': 1e-4}

last_layer_optimizer_lr = 1e-4
last_layer_fixed = True

coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.08,
    'l1': 1e-2,
    'offset_bias_l2': 8e-1,
    'offset_weight_l2': 8e-1,
    'orthogonality_loss': 0.1
}

subtractive_margin = True

num_train_epochs = 100
num_warm_epochs = 5
num_secondary_warm_epochs = 5
push_start = 20

push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
