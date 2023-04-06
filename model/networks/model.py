# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch.nn.functional as F
from networks.layer import MLP
from networks.module import BackboneNet, DecoderNet, DecoderNet_big
from networks.transformer import Transformer
#from common.nets.loss import *
from networks.position_encoding import build_position_encoding
from networks.preprocessing import PeakDetector
#from common.utils.misc import *
import numpy as np
import time
import torch.nn as nn
import torch
from .config import cfg


class Model(nn.Module):
    def __init__(self, backbone_net, decoder_net, transformer, position_embedding, hidden_dim=256,mutliscale_dim =992,output_hm_shape=[256,256]):
        super(Model, self).__init__()
        output_dim = hidden_dim - 32
        self.output_hm_shape = output_hm_shape
        # modules
        self.backbone_net = backbone_net
        self.decoder_net = decoder_net
        self.transformer = transformer
        self.position_embedding = position_embedding

        start = time.time()
        self.peak_detector = PeakDetector()
        self.obj_peak_detector = PeakDetector(nearest_neighbor_th=5)
        print('Init of peak detector took %f s'%(time.time()-start))

        # MLP for converting concatenated image features to 256-D features
        self.norm1 = nn.LayerNorm(mutliscale_dim)
        self.linear1 = MLP(input_dim = mutliscale_dim, hidden_dim=[1024, 512, 256],
                           output_dim=output_dim,
                           num_layers=4, is_activation_last=True)
        self.activation = nn.functional.relu
        self.sigmod = nn.Sigmoid()
        # MLP for predicting the hand type after the U-Net decoder
        self.linear_class = MLP(256, 256, 128, 2)


        # Freeze batch norm layers
        self.freeze_stages()
     

    def freeze_stages(self):

        for name, param in self.backbone_net.named_parameters():
            if 'bn' in name:
                param.requires_grad = False

    def get_input_seq(self, joint_heatmap_out, feature_pyramid, pos_embed, epoch_cnt):
        heatmap_np = joint_heatmap_out.detach().cpu().numpy()

        grids = []
        masks = []
        max_num_peaks = 1
        peak_joints_map_batch = []

        normalizer = np.array([self.output_hm_shape[0] - 1, self.output_hm_shape[1] - 1]) / 2
        for ii in range(heatmap_np.shape[0]):
            map_slics = []
            mask_slices = []
            for m_id in range(heatmap_np.shape[1]):

                peaks, peaks_ind_list = self.peak_detector.detect_peaks_nms(heatmap_np[ii,m_id],
                                                                            max_num_peaks)
                
                #print("peaks",peaks.shape, pos_embed.shape)
                peak_joints_map = np.zeros((len(peaks_ind_list)), dtype=int)+1
    
                if len(peak_joints_map) != len(peaks_ind_list):
                    print(len(peak_joints_map), len(peaks_ind_list))
                    assert len(peak_joints_map) == len(peaks_ind_list)
    
                if len(peaks_ind_list) == 0:
                    # Corner case when the object and hand is heavily occluded in the image
    
                    # print('Found %d peaks for %s/%s/%s/%s'%(len(peaks_ind_list), str(meta_info['capture'][ii]),
                    #                                      str(meta_info['cam'][ii]), str(meta_info['seq_id'][ii]), str(meta_info['frame'][ii])))
    
                    peaks_pixel_locs_normalized = np.tile(np.array([[-1, -1]]), (max_num_peaks,1))
                    mask = np.ones((max_num_peaks), dtype= bool)
                    peak_joints_map = np.zeros((max_num_peaks,), dtype=int)
                else:
                    peaks_ind_normalized = (np.array(peaks_ind_list) - normalizer) / normalizer
                    assert np.sum(peaks_ind_normalized < -1) == 0 and np.sum(peaks_ind_normalized > 1) == 0
    
                    peaks_pixel_locs_normalized = peaks_ind_normalized[:, [1, 0]]  # in pixel coordinates
                    mask = np.ones((peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)
    
                    # fill up the empty slots with some dummy values
                    if peaks_pixel_locs_normalized.shape[0] < max_num_peaks:
                        dummy_peaks = np.tile(np.array([[-1, -1]]), (max_num_peaks - peaks_pixel_locs_normalized.shape[0],1))
                        invalid_mask = np.zeros((max_num_peaks - peaks_pixel_locs_normalized.shape[0],), dtype=np.bool)
                        peak_joints_map = np.concatenate([peak_joints_map, invalid_mask.astype(np.int)], axis=0)
    
                        peaks_pixel_locs_normalized = np.concatenate([peaks_pixel_locs_normalized, dummy_peaks], axis=0)
                        mask = np.concatenate([mask, invalid_mask], axis=0)
                map_slics.append(peaks_pixel_locs_normalized)
                mask_slices.append(mask)
            #print("peaks_pixel_locs_normalized",peaks_pixel_locs_normalized.shape)
            #print(len(map_slics), np.concatenate(map_slics, 0).shape)
            grids.append(np.concatenate(map_slics, 0))
            masks.append(np.concatenate(mask_slices, 0))
            peak_joints_map_batch.append(peak_joints_map)

        #print(len(grids))
        peak_joints_map_batch = torch.from_numpy(np.stack(peak_joints_map_batch, 0)).to(pos_embed.device) # N x max_num_peaks
        grids = np.stack(grids, 0)  # N x max_num_peaks x 2
        #print(grids.shape)
        grids_unnormalized_np = grids*normalizer[[1,0]] + normalizer[[1,0]] # in pixel coordinates space
        masks_np = np.stack(masks, 0)  # N x max_num_peaks
        masks = torch.from_numpy(masks_np).bool().to(pos_embed.device) # N x max_num_peaks

        # Get the positional embeddings
        positions = nn.functional.grid_sample(pos_embed,
                                              torch.from_numpy(np.expand_dims(grids, 1)).float().to(pos_embed.device),
                                              mode='nearest', align_corners=True).squeeze(2) # N x hidden_dim x max_num_peaks

        # Sample the CNN features
        multiscale_features = []
        grids_tensor = torch.from_numpy(np.expand_dims(grids, 1)).float().to(feature_pyramid[cfg.mutliscale_layers[0]].device)
        for layer_name in cfg.mutliscale_layers:
            # N x C x 1 x max_num_peaks
            multiscale_features.append(torch.nn.functional.grid_sample(feature_pyramid[layer_name],
                                                                       grids_tensor,
                                                                         align_corners=True))


        #print('!!!!!', multiscale_features.shape)
        multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2) # N x C1 x  max_num_peaks
        print('!!!!!', multiscale_features.shape)
        multiscale_features = multiscale_features.permute(0, 2, 1) # N x max_num_peaks x C1


        print(multiscale_features.shape)
        input_seq = self.linear1(multiscale_features).permute(0, 2, 1) # N x hidden_dim x max_num_peaks

        return input_seq, masks, positions, grids_unnormalized_np, masks_np, peak_joints_map_batch


    def forward(self, inputs, epoch_cnt=1e8):
        input_img = inputs['img']
        input_mask = inputs['mask']
        batch_size = input_img.shape[0]

        img_feat, enc_skip_conn_layers = self.backbone_net(input_img)
        feature_pyramid, decoder_out = self.decoder_net(img_feat, enc_skip_conn_layers)
        
        print(decoder_out.shape)
        #print(feature_pyramid)
        # for fe in feature_pyramid:
        #     print(fe.shape)
        
        joint_heatmap_out = decoder_out#[:,0]


        # Get the positional embeddings
        pos = self.position_embedding(nn.functional.interpolate(input_img, (cfg.output_hm_shape[2], cfg.output_hm_shape[1])),
                                nn.functional.interpolate(input_mask, (cfg.output_hm_shape[2], cfg.output_hm_shape[1])))

        print('~~~~',pos.shape, joint_heatmap_out.shape)
        # Get the input tokens
        input_seq, masks, positions, joint_loc_pred_np, mask_np, peak_joints_map_batch \
            = self.get_input_seq(joint_heatmap_out, feature_pyramid, pos, epoch_cnt)


        print("input_seq, masks",input_seq.shape, positions.shape)
        # Concatenate positional and appearance embeddings
        if cfg.position_embedding == 'simpleCat':
            input_seq = torch.cat([input_seq, positions], dim=1)
            print(input_seq.shape)
            positions = torch.zeros_like(input_seq).to(input_seq.device)

        #transformer_out, hand_type, memory, encoder_out, attn_wts_all_layers 
        encoder_out0, encoder_out = self.transformer(src=input_seq, mask=torch.logical_not(masks), pos_embed=positions)
        encoder_out0 = encoder_out0.permute(1, 0, 2)
        print("encoder_out0, encoder_out", encoder_out0.shape, encoder_out.shape)
        # Make all the predictions
        joint_class = self.linear_class(encoder_out0) # 6 x max_num_peaks x N x 22(43)
        print(joint_class.shape)
        # out['hand_type_out'] = torch.argmax(hand_type[-1,0], dim=1) # N

        # loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap, meta_info['joint_valid'], meta_info['hm_valid'])

        # Put all the outputs in a dict
        # out = {}
        # out1 = {**loss, **out}
        return self.sigmod(joint_heatmap_out), joint_class



def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode, joint_num):
    backbone_net = BackboneNet()

    if cfg.use_big_decoder:
        decoder_net = DecoderNet_big(joint_num)
    else:
        decoder_net = DecoderNet(joint_num)

    transformer = Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

    print('BackboneNet No. of Params = %d'%(sum(p.numel() for p in backbone_net.parameters() if p.requires_grad)))
    print('decoder_net No. of Params = %d' % (sum(p.numel() for p in decoder_net.parameters() if p.requires_grad)))
    print('transformer No. of Params = %d' % (sum(p.numel() for p in transformer.parameters() if p.requires_grad)))


    position_embedding = build_position_encoding(cfg)

    if mode == 'train':
        backbone_net.init_weights()
        decoder_net.apply(init_weights)

    model = Model(backbone_net, decoder_net, transformer, position_embedding)
    print('Total No. of Params = %d' % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    return model

if __name__ == "__main__":
    model = get_model('train', 12)
    a = torch.zeros((2, 3, 256, 256))#.cuda()    
    b = torch.zeros((2, 1, 256, 256))#.cuda()
    c = model({"img":a, "mask":b})
    print(c.shape)