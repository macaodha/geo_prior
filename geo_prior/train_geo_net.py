import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch

import models
import utils as ut
import datasets as dt
import grid_predictor as grid
from paths import get_paths
import losses as lo


class LocationDataLoader(torch.utils.data.Dataset):
    def __init__(self, loc_feats, labels, users, num_classes, is_train):
        self.loc_feats = loc_feats
        self.labels = labels
        self.users = users
        self.is_train = is_train
        self.num_classes = num_classes


    def __len__(self):
        return len(self.loc_feats)

    def __getitem__(self, index):
        loc_feat  = self.loc_feats[index, :]
        loc_class = self.labels[index]
        user      = self.users[index]
        if self.is_train:
            return loc_feat, loc_class, user
        else:
            return loc_feat, loc_class


def generate_feats(locs, dates, params):
    x_locs = ut.convert_loc_to_tensor(locs, params['device'])
    x_dates = torch.from_numpy(dates.astype(np.float32)*2 - 1).to(params['device'])
    feats = ut.encode_loc_time(x_locs, x_dates, concat_dim=1, params=params)
    return feats


def train(model, data_loader, optimizer, epoch, params):
    model.train()

    # adjust the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = params['lr'] * (params['lr_decay'] ** epoch)

    loss_avg = ut.AverageMeter()
    inds = torch.arange(params['batch_size']).to(params['device'])

    for batch_idx, (loc_feat, loc_class, user_ids) in enumerate(data_loader):
        optimizer.zero_grad()

        loss = lo.embedding_loss(model, params, loc_feat, loc_class, user_ids, inds)

        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item(), len(loc_feat))

        if (batch_idx % params['log_frequency'] == 0 and batch_idx != 0) or (batch_idx == (len(data_loader)-1)):
            print('[{}/{}]\tLoss  : {:.4f}'.format(batch_idx * params['batch_size'], len(data_loader.dataset), loss_avg.avg))


def test(model, data_loader, params):
    # NOTE the test loss only tracks the BCE it is not the full loss used during training
    model.eval()
    loss_avg = ut.AverageMeter()

    inds = torch.arange(params['batch_size']).to(params['device'])
    with torch.no_grad():

        for loc_feat, loc_class in data_loader:

            loc_pred = model(loc_feat)
            pos_loss = lo.bce_loss(loc_pred[inds[:loc_feat.shape[0]], loc_class])
            loss = pos_loss.mean()

            loss_avg.update(loss.item(), loc_feat.shape[0])

    print('Test loss   : {:.4f}'.format(loss_avg.avg))


def plot_gt_locations(params, mask, train_classes, class_of_interest, classes, train_locs, train_dates, op_dir):
    # plot GT locations for the class of interest
    im_width  = (params['map_range'][1] - params['map_range'][0]) // 45  # 8
    im_height = (params['map_range'][3] - params['map_range'][2]) // 45  # 4
    plt.figure(num=0, figsize=[im_width, im_height])
    plt.imshow(mask, extent=params['map_range'], cmap='tab20')

    inds = np.where(train_classes==class_of_interest)[0]
    print('{} instances of: '.format(len(inds)) + classes[class_of_interest])

    # the color of the dot indicates the date
    colors = np.sin(np.pi*train_dates[inds])
    plt.scatter(train_locs[inds, 0], train_locs[inds, 1], c=colors, s=2, cmap='magma', vmin=0, vmax=1)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)

    op_file_name = op_dir + 'gt_' + str(class_of_interest).zfill(4) + '.jpg'
    plt.savefig(op_file_name, dpi=400, bbox_inches='tight',pad_inches=0)


def main():

    # hyper params
    params = {}
    params['dataset'] = 'inat_2018'  # inat_2018, inat_2017, birdsnap, nabirds, yfcc
    if params['dataset'] in ['birdsnap', 'nabirds']:
        params['meta_type'] = 'ebird_meta'  # orig_meta, ebird_meta
    else:
        params['meta_type'] = ''
    params['batch_size'] = 1024
    params['lr'] = 0.0005
    params['lr_decay'] = 0.98
    params['num_filts'] = 256  # embedding dimension
    params['num_epochs'] = 30
    params['log_frequency'] = 50
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    params['balanced_train_loader'] = True
    params['max_num_exs_per_class'] = 100
    params['map_range'] = (-180, 180, -90, 90)

    # specify feature encoding for location and date
    params['use_date_feats'] = True  # if False date feature is not used
    params['loc_encode']     = 'encode_cos_sin'  # encode_cos_sin, encode_3D, encode_none
    params['date_encode']    = 'encode_cos_sin' # encode_cos_sin, encode_none

    # specify loss type
    # appending '_user' models the user location and object affinity - see losses.py
    params['train_loss'] = 'full_loss_user'  # full_loss_user, full_loss

    print('Dataset   \t' + params['dataset'])
    op = dt.load_dataset(params, 'val', True, True)
    train_locs = op['train_locs']
    train_classes = op['train_classes']
    train_users = op['train_users']
    train_dates = op['train_dates']
    val_locs = op['val_locs']
    val_classes = op['val_classes']
    val_users = op['val_users']
    val_dates = op['val_dates']
    class_of_interest = op['class_of_interest']
    classes = op['classes']
    params['num_classes'] = op['num_classes']

    if params['meta_type'] == '':
        params['model_file_name'] = '../models/model_' + params['dataset'] + '.pth.tar'
    else:
        params['model_file_name'] = '../models/model_' + params['dataset'] + '_' + params['meta_type'] + '.pth.tar'
    op_dir = 'ims/ims_' + params['dataset'] + '/'
    if not os.path.isdir(op_dir):
        os.makedirs(op_dir)

    # process users
    # NOTE we are only modelling the users in the train set - not the val
    un_users, train_users = np.unique(train_users, return_inverse=True)
    train_users = torch.from_numpy(train_users).to(params['device'])
    params['num_users'] = len(un_users)
    if 'user' in params['train_loss']:
        assert (params['num_users'] != 1)  # need to have more than one user

    # print stats
    print('\nnum_classes\t{}'.format(params['num_classes']))
    print('num train    \t{}'.format(len(train_locs)))
    print('num val      \t{}'.format(len(val_locs)))
    print('train loss   \t' + params['train_loss'])
    print('model name   \t' + params['model_file_name'])
    print('num users    \t{}'.format(params['num_users']))
    if params['meta_type'] != '':
        print('meta data    \t' + params['meta_type'])

    # load ocean mask for plotting
    mask = np.load(get_paths('mask_dir') + 'ocean_mask.npy').astype(np.int)

    # data loaders
    train_labels = torch.from_numpy(train_classes).to(params['device'])
    train_feats = generate_feats(train_locs, train_dates, params)
    train_dataset = LocationDataLoader(train_feats, train_labels, train_users, params['num_classes'], True)
    if params['balanced_train_loader']:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'],
                       sampler=ut.BalancedSampler(train_classes.tolist(), params['max_num_exs_per_class'],
                       use_replace=False, multi_label=False), shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=True)

    val_labels = torch.from_numpy(val_classes).to(params['device'])
    val_feats = generate_feats(val_locs, val_dates, params)
    val_dataset = LocationDataLoader(val_feats, val_labels, val_users, params['num_classes'], False)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=params['batch_size'], shuffle=False)

    # create model
    params['num_feats'] = train_feats.shape[1]
    model = models.FCNet(num_inputs=params['num_feats'], num_classes=params['num_classes'],
                         num_filts=params['num_filts'], num_users=params['num_users']).to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # set up grid to make dense prediction across world
    gp = grid.GridPredictor(mask, params)

    # plot ground truth
    plt.close('all')
    plot_gt_locations(params, mask, train_classes, class_of_interest, classes, train_locs, train_dates, op_dir)


    # main train loop
    for epoch in range(0, params['num_epochs']):
        print('\nEpoch\t{}'.format(epoch))
        train(model, train_loader, optimizer, epoch, params)
        test(model, val_loader, params)

        # save dense prediction image
        grid_pred = gp.dense_prediction(model, class_of_interest)
        op_file_name = op_dir + str(epoch).zfill(4) + '_' + str(class_of_interest).zfill(4) + '.jpg'
        plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)


    if params['use_date_feats']:
        print('\nGenerating predictions for each month of the year.')
        if not os.path.isdir(op_dir + 'time/'):
            os.makedirs(op_dir + 'time/')
        for ii, tm in enumerate(np.linspace(0,1,13)):
           grid_pred = gp.dense_prediction(model, class_of_interest, tm)
           op_file_name = op_dir + 'time/' + str(class_of_interest).zfill(4) + '_' + str(ii) + '.jpg'
           plt.imsave(op_file_name, 1-grid_pred, cmap='afmhot', vmin=0, vmax=1)


    # save trained model
    print('Saving output model to ' + params['model_file_name'])
    op_state = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'params' : params}
    torch.save(op_state, params['model_file_name'])


if __name__== "__main__":
    main()