import numpy as np
import torch
import json
import os
import math
from torch.utils.data.sampler import Sampler


def encode_loc_time(loc_ip, date_ip, concat_dim=1, params=None):
    # assumes inputs location and date features are in range -1 to 1
    # location is lon, lat

    if params['loc_encode'] == 'encode_cos_sin':
        feats = torch.cat((torch.sin(math.pi*loc_ip), torch.cos(math.pi*loc_ip)), concat_dim)

    elif params['loc_encode'] == 'encode_3D':
        # X, Y, Z in 3D space
        if concat_dim == 1:
            cos_lon = torch.cos(math.pi*loc_ip[:, 0]).unsqueeze(-1)
            sin_lon = torch.sin(math.pi*loc_ip[:, 0]).unsqueeze(-1)
            cos_lat = torch.cos(math.pi*loc_ip[:, 1]).unsqueeze(-1)
            sin_lat = torch.sin(math.pi*loc_ip[:, 1]).unsqueeze(-1)
        if concat_dim == 2:
            cos_lon = torch.cos(math.pi*loc_ip[:, :, 0]).unsqueeze(-1)
            sin_lon = torch.sin(math.pi*loc_ip[:, :, 0]).unsqueeze(-1)
            cos_lat = torch.cos(math.pi*loc_ip[:, :, 1]).unsqueeze(-1)
            sin_lat = torch.sin(math.pi*loc_ip[:, :, 1]).unsqueeze(-1)
        feats = torch.cat((cos_lon*cos_lat, sin_lon*cos_lat, sin_lat), concat_dim)

    elif params['loc_encode'] == 'encode_none':
        feats = loc_ip

    else:
        print('error - no loc feat type defined')


    if params['use_date_feats']:
        if params['date_encode'] == 'encode_cos_sin':
            feats_date = torch.cat((torch.sin(math.pi*date_ip.unsqueeze(-1)),
                                    torch.cos(math.pi*date_ip.unsqueeze(-1))), concat_dim)
        elif params['date_encode'] == 'encode_none':
            feats_date = date_ip.unsqueeze(-1)
        else:
            print('error - no date feat type defined')
        feats = torch.cat((feats, feats_date), concat_dim)

    return feats


class BalancedSampler(Sampler):
    # sample "evenly" from each from class
    def __init__(self, classes, num_per_class, use_replace=False, multi_label=False):
        self.class_dict = {}
        self.num_per_class = num_per_class
        self.use_replace = use_replace
        self.multi_label = multi_label

        if self.multi_label:
            self.class_dict = classes
        else:
            # standard classification
            un_classes = np.unique(classes)
            for cc in un_classes:
                self.class_dict[cc] = []

            for ii in range(len(classes)):
                self.class_dict[classes[ii]].append(ii)

        if self.use_replace:
            self.num_exs = self.num_per_class*len(un_classes)
        else:
            self.num_exs = 0
            for cc in self.class_dict.keys():
                self.num_exs += np.minimum(len(self.class_dict[cc]), self.num_per_class)


    def __iter__(self):
        indices = []
        for cc in self.class_dict:
            if self.use_replace:
                indices.extend(np.random.choice(self.class_dict[cc], self.num_per_class).tolist())
            else:
                indices.extend(np.random.choice(self.class_dict[cc], np.minimum(len(self.class_dict[cc]),
                                                self.num_per_class), replace=False).tolist())
        # in the multi label setting there will be duplictes at training time
        np.random.shuffle(indices)  # will remain a list
        return iter(indices)

    def __len__(self):
        return self.num_exs


def convert_loc_to_tensor(x, device=None):
    # intput is in lon {-180, 180}, lat {90, -90}
    xt = x.astype(np.float32)
    xt[:,0] /= 180.0
    xt[:,1] /= 90.0
    xt = torch.from_numpy(xt)
    if device is not None:
        xt = xt.to(device)
    return xt


def distance_pw_euclidean(xx, yy):
    # equivalent to scipy.spatial.distance.cdist
    dist = np.sqrt((xx**2).sum(1)[:, np.newaxis] - 2*xx.dot(yy.transpose()) + ((yy**2).sum(1)[np.newaxis, :]))
    return dist


def distance_pw_haversine(xx, yy, radius=6372.8):
    # input should be in radians
    # output is in km's if radius = 6372.8

    d_lon = xx[:, 0][..., np.newaxis] - yy[:, 0][np.newaxis, ...]
    d_lat = xx[:, 1][..., np.newaxis] - yy[:, 1][np.newaxis, ...]

    cos_term = np.cos(xx[:,1])[..., np.newaxis]*np.cos(yy[:, 1])[np.newaxis, ...]
    dist = np.sin(d_lat/2.0)**2 + cos_term*np.sin(d_lon/2.0)**2
    dist = 2 * radius * np.arcsin(np.sqrt(dist))
    return dist


def euclidean_distance(xx, yy):
    return np.sqrt(((xx - yy)**2).sum(1))


def haversine_distance(xx, yy, radius=6371.4):
    # assumes shape N x 2, where col 0 is lat, and col 1 is lon
    # input should be in radians
    # output is in km's if radius = 6371.4
    # note that SKLearns haversine distance is [latitude, longitude] not [longitude, latitude]

    d_lon = xx[:, 0] - yy[0]
    d_lat = xx[:, 1] - yy[1]

    cos_term = np.cos(xx[:,1])*np.cos(yy[1])
    dist = np.sin(d_lat/2.0)**2 + cos_term*np.sin(d_lon/2.0)**2
    dist = 2 * radius * np.arcsin(np.sqrt(dist + 1e-16))

    return dist


def bilinear_interpolate(loc, data, remove_nans=False):
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    # map to [0,1], then scale to data size
    loc = (loc.clone() + 1) / 2.0
    if remove_nans:
        loc[torch.isnan(loc)] = 0.5
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx.clone() + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy.clone() + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val


class AverageMeter:
    # computes and stores the average and current value

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / float(self.count)
