import torch
import utils as ut
import math


def bce_loss(pred):
    return -torch.log(pred + 1e-5)


def rand_samples(batch_size, params, rand_type='uniform'):
    # randomly sample background locations
    rand_feats_orig = torch.rand(batch_size, 3).to(params['device'])*2 -1

    if rand_type == 'spherical':
        theta = ((rand_feats_orig[:,1].unsqueeze(1)+1) / 2.0)*(2*math.pi)
        r_lon = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.cos(theta)
        r_lat = torch.sqrt(1.0 - rand_feats_orig[:,0].unsqueeze(1)**2) * torch.sin(theta)
        rand_feats_orig = torch.cat((r_lon, r_lat, rand_feats_orig[:,2].unsqueeze(1)), 1)

    rand_feats = ut.encode_loc_time(rand_feats_orig[:,:2], rand_feats_orig[:,2], concat_dim=1, params=params)
    return rand_feats


def embedding_loss(model, params, loc_feat, loc_class, user_ids, inds):

    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    # create random background samples
    loc_feat_rand = rand_samples(batch_size, params, rand_type='spherical')

    # get location embeddings
    loc_cat = torch.cat((loc_feat, loc_feat_rand), 0)
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]

    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))

    # data loss
    pos_weight = params['num_classes']
    loss_pos = bce_loss(1.0 - loc_pred)  # neg
    loss_pos[inds[:batch_size], loc_class] = pos_weight*bce_loss(loc_pred[inds[:batch_size], loc_class])  # pos
    loss_bg = bce_loss(1.0 - loc_pred_rand)

    if 'user' in params['train_loss']:

        # user location loss
        user = model.user_emb.weight[user_ids, :]
        p_u_given_l = torch.sigmoid((user*loc_emb).sum(1))
        p_u_given_randl = torch.sigmoid((user*loc_emb_rand).sum(1))

        user_loc_pos_loss = bce_loss(p_u_given_l)
        user_loc_neg_loss = bce_loss(1.0 - p_u_given_randl)

        # user class loss
        p_c_given_u = torch.sigmoid(torch.matmul(user, model.class_emb.weight.transpose(0,1)))
        user_class_loss = bce_loss(1.0 - p_c_given_u)
        user_class_loss[inds[:batch_size], loc_class] = pos_weight*bce_loss(p_c_given_u[inds[:batch_size], loc_class])

        # total loss
        loss = loss_pos.mean() + loss_bg.mean() + user_loc_pos_loss.mean() + \
               user_loc_neg_loss.mean() + user_class_loss.mean()

    else:

        # total loss
        loss = loss_pos.mean() + loss_bg.mean()

    return loss
