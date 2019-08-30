"""
Class for making dense predictions on a 2D grid.
"""

import numpy as np
import torch
import math
import sys
sys.path.append('../')
import geo_prior.utils as ut


class GridPredictor:

    def __init__(self, mask, params, mask_only_pred=False):
        # set up coordinates to make dense prediction on grid
        self.device = params['device']
        self.params = params
        self.mask = mask
        self.use_date_feats = params['use_date_feats']

        self.mask_lines = (np.gradient(mask)[0]**2 + np.gradient(mask)[1]**2)
        self.mask_lines[self.mask_lines > 0.0] = 1.0

        # set up feature grid this will be height X width X num feats
        grid_lon = torch.linspace(-1, 1, mask.shape[1]).to(self.device)
        grid_lon = grid_lon.repeat(mask.shape[0],1).unsqueeze(2)
        grid_lat = torch.linspace(1, -1, mask.shape[0]).to(self.device)
        grid_lat = grid_lat.repeat(mask.shape[1], 1).transpose(0,1).unsqueeze(2)
        dates  = torch.zeros(mask.shape[0], mask.shape[1], 1, device=self.device)

        if self.use_date_feats:
            loc_time_feats = torch.cat((grid_lon, grid_lat, dates), 2)
            loc_time_feats = ut.encode_loc_time(loc_time_feats[:,:,:2], loc_time_feats[:,:,2], concat_dim=2, params=params)
        else:
            loc_time_feats = torch.cat((grid_lon, grid_lat), 2)
            loc_time_feats = ut.encode_loc_time(loc_time_feats[:,:,:2], None, concat_dim=2, params=params)
        self.feats = loc_time_feats

        # for mask only prediction
        if mask_only_pred:
            self.mask_inds = np.where(self.mask.ravel() == 1)[0]
            self.feats_local = self.feats.reshape(self.feats.shape[0]*self.feats.shape[1], self.feats.shape[2])[self.mask_inds, :].clone()


    def dense_prediction(self, model, class_of_interest, time_step=0, mask_op=True):
        # make prediction for entire grid at different time steps - by looping over columns
        # time_step should be in range [0, 1]

        feats_change_time = self.feats.clone()
        if self.use_date_feats:
            feats_change_time = self.update_date_feats(feats_change_time, time_step)

        grid_pred = np.zeros(self.mask.shape, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for col in range(feats_change_time.shape[1]):
                pred = model(feats_change_time[:,col,:], class_of_interest=class_of_interest)
                grid_pred[:, col] = pred.cpu().numpy()

        if mask_op:
            return grid_pred*self.mask + self.mask_lines
        else:
            return grid_pred

    def dense_prediction_sum(self, model, time_step=0, mask_op=True):
        # make prediction for entire grid at different time steps - by looping over columns
        # takes the mean prediction for each class

        feats_change_time = self.feats.clone()
        if self.use_date_feats:
            feats_change_time = self.update_date_feats(feats_change_time, time_step)

        grid_pred = np.zeros(self.mask.shape, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for col in range(feats_change_time.shape[1]):
                pred = model(feats_change_time[:,col,:]).sum(1)
                grid_pred[:, col] = pred.cpu().numpy()

        max_val = grid_pred.max()
        if mask_op:
            return grid_pred*self.mask + self.mask_lines, max_val
        else:
            return grid_pred, max_val

    def dense_prediction_masked(self, model, class_of_interest, time_step):
        # only masks predictions for valid datapoints
        if self.use_date_feats:
            self.feats_local = self.update_date_feats(feats_local, time_step)

        model.eval()
        with torch.no_grad():
            pred = model(self.feats_local, class_of_interest=class_of_interest)

        grid_pred = self.create_full_output(self, pred.cpu().numpy())
        return grid_pred

    def dense_prediction_masked_feats(self, model, time_step):
        # only masks predictions for valid datapoints
        if self.use_date_feats:
            feats_local = self.update_date_feats(self.feats_local, time_step)

        model.eval()
        with torch.no_grad():
            feats = model(self.feats_local, return_feats=True)

        return feats

    def create_full_output(self, pred):
        grid_pred = np.zeros(self.mask.shape[0]*self.mask.shape[1], dtype=np.float32)
        grid_pred[self.mask_inds] = pred
        return grid_pred.reshape((self.mask.shape[0], self.mask.shape[1]))


    def update_date_feats(self, feats, time_step):
        # helper function - for visualization we want to vary the date
        offset = 0
        if self.params['loc_encode'] == 'encode_cos_sin':
            offset = 4
        elif self.params['loc_encode'] == 'encode_3D':
            offset = 3

        if len(feats.shape) == 2:
            if self.params['date_encode'] == 'encode_cos_sin':
                feats[:,offset]   = math.sin(math.pi*(2*time_step - 1))
                feats[:,offset+1] = math.cos(math.pi*(2*time_step - 1))
            elif self.params['date_encode'] == 'encode_none':
                feats[:,offset]   = (2*time_step - 1)
        else:
            if self.params['date_encode'] == 'encode_cos_sin':
                feats[:,:,offset]   = math.sin(math.pi*(2*time_step - 1))
                feats[:,:,offset+1] = math.cos(math.pi*(2*time_step - 1))
            elif self.params['date_encode'] == 'encode_none':
                feats[:,:,offset]   = (2*time_step - 1)

        return feats
