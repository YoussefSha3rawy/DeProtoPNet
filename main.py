import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import argparse
import re

from DeformableProtoPNet.helpers import makedir
from DeformableProtoPNet import model, push, train_and_test as tnt, save
from DeformableProtoPNet.log import create_logger
from DeformableProtoPNet.preprocess import mean, std, preprocess_input_function
from logger import WandbLogger

"""
python3 main.py -gpuid='0, 1' \
                    -m=0.1 \
                    -last_layer_fixed=True \
                    -subtractive_margin=True \
                    -using_deform=True \
                    -topk_k=1 \
                    -num_prototypes=1200 \
                    -incorrect_class_connection=-0.5 \
                    -deformable_conv_hidden_channels=128 \
                    -rand_seed=1
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    # parser.add_argument('-m', nargs=1, type=float, default=None)
    # parser.add_argument('-last_layer_fixed', nargs=1, type=str, default=None)
    # parser.add_argument('-subtractive_margin', nargs=1, type=str, default=None)
    # parser.add_argument('-using_deform', nargs=1, type=str, default=None)
    # parser.add_argument('-topk_k', nargs=1, type=int, default=None)
    # parser.add_argument('-deformable_conv_hidden_channels', nargs=1, type=int, default=None)
    # parser.add_argument('-num_prototypes', nargs=1, type=int, default=None)
    # parser.add_argument('-dilation', nargs=1, type=float, default=2)
    # parser.add_argument('-incorrect_class_connection', nargs=1, type=float, default=0)
    # parser.add_argument('-rand_seed', nargs=1, type=int, default=None)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    m = 0.1
    rand_seed = 1
    last_layer_fixed = True
    subtractive_margin = True
    using_deform = True
    topk_k = 1
    deformable_conv_hidden_channels = 128
    dilation = 2
    incorrect_class_connection = -0.5

    print("---- USING DEFORMATION: ", using_deform)
    print("Margin set to: ", m)
    print("last_layer_fixed set to: {}".format(last_layer_fixed))
    print("subtractive_margin set to: {}".format(subtractive_margin))
    print("topk_k set to: {}".format(topk_k))
    print("num_prototypes set to: {}".format(num_prototypes))
    print("incorrect_class_connection: {}".format(incorrect_class_connection))
    print("deformable_conv_hidden_channels: {}".format(deformable_conv_hidden_channels))

    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    print("Random seed: ", rand_seed)
        
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    from config import img_size, experiment_run, base_architecture, num_prototypes

    if 'resnet34' in base_architecture:
        prototype_shape = (num_prototypes, 512, 2, 2)
        add_on_layers_type = 'upsample'
    elif 'resnet152' in base_architecture:
        prototype_shape = (num_prototypes, 2048, 2, 2)
        add_on_layers_type = 'upsample'
    elif 'resnet50' in base_architecture:
        prototype_shape = (num_prototypes, 2048, 2, 2)
        add_on_layers_type = 'upsample'
    elif 'densenet121' in base_architecture:
        prototype_shape = (num_prototypes, 1024, 2, 2)
        add_on_layers_type = 'upsample'
    elif 'densenet161' in base_architecture:
        prototype_shape = (num_prototypes, 2208, 2, 2)
        add_on_layers_type = 'upsample'
    else:
        prototype_shape = (num_prototypes, 512, 2, 2)
        add_on_layers_type = 'upsample'
    print("Add on layers type: ", add_on_layers_type)


    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

    from config import train_dir, test_dir, train_push_dir

    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'config.py'), dst=model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    wandb_logger = WandbLogger(
            {'base_architecture': base_architecture, 'experiment_run': experiment_run}, logger_name='DeProtoPNet', project='FinalProject')
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    from config import train_batch_size, test_batch_size, train_push_batch_size, num_classes

    normalize = transforms.Normalize(mean=mean,
                                    std=std)

    if 'augmented' not in train_dir:
        print("Using online augmentation")
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomAffine(degrees=(-25, 25), shear=15),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=8, pin_memory=False)
    # push set
    train_push_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=8, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('test set size: {0}'.format(len(test_loader.dataset)))
    log('batch size: {0}'.format(train_batch_size))

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes, topk_k=topk_k, m=m,
                                add_on_layers_type=add_on_layers_type,
                                using_deform=using_deform,
                                incorrect_class_connection=incorrect_class_connection,
                                deformable_conv_hidden_channels=deformable_conv_hidden_channels,
                                prototype_dilation=2)
        
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # define optimizer
    from config import joint_optimizer_lrs, joint_lr_step_size
    if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
        joint_optimizer_lrs['features'] = 1e-5
    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    {'params': ppnet.conv_offset.parameters(), 'lr': joint_optimizer_lrs['conv_offset']},
    {'params': ppnet.last_layer.parameters(), 'lr': joint_optimizer_lrs['joint_last_layer_lr']}
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
    log("joint_optimizer_lrs: ")
    log(str(joint_optimizer_lrs))

    from config import warm_optimizer_lrs
    warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    log("warm_optimizer_lrs: ")
    log(str(warm_optimizer_lrs))

    from config import warm_pre_offset_optimizer_lrs
    if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
        warm_pre_offset_optimizer_lrs['features'] = 1e-5
    warm_pre_offset_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_pre_offset_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': warm_pre_offset_optimizer_lrs['prototype_vectors']},
    {'params': ppnet.features.parameters(), 'lr': warm_pre_offset_optimizer_lrs['features'], 'weight_decay': 1e-3},
    ]
    warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)

    warm_lr_scheduler = None
    if 'stanford_dogs' in train_dir:
        warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
        log("warm_pre_offset_optimizer_lrs: ")
        log(str(warm_pre_offset_optimizer_lrs))

    from config import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    # weighting of different training losses
    from config import coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    from config import num_warm_epochs, num_train_epochs, push_epochs, \
                        num_secondary_warm_epochs, push_start

    # train the model
    log('start training')

    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                        use_ortho_loss=False)
        elif epoch >= num_warm_epochs and epoch - num_warm_epochs < num_secondary_warm_epochs:
            tnt.warm_pre_offset(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_pre_offset_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                        use_ortho_loss=False)
            if 'stanford_dogs' in train_dir:
                warm_lr_scheduler.step()
        else:
            if epoch == num_warm_epochs + num_secondary_warm_epochs:
                ppnet_multi.module.initialize_offset_weights()
            tnt.joint(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                        use_ortho_loss=True)
            joint_lr_scheduler.step()

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log, subtractive_margin=subtractive_margin)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.70, log=log)

        if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
            push.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                        target_accu=0.70, log=log)

            if not last_layer_fixed:
                tnt.last_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log, 
                                subtractive_margin=subtractive_margin)
                    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                                target_accu=0.70, log=log)
    logclose()

if __name__ == "__main__":
    main()