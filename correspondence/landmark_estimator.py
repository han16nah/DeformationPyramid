from multiprocessing.util import debug
import torch
import yaml
from easydict import EasyDict as edict


import sys
sys.path.append("")
from lepard.pipeline import Pipeline as Matcher
from outlier_rejection.pipeline import   Outlier_Rejection
from outlier_rejection.loss import   NeCoLoss



class Landmark_Model ():

    def __init__(self, config_file, device ):

        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = edict(config)

        with open(config['matcher_config'], 'r') as f_:
            matcher_config = yaml.load(f_, Loader=yaml.Loader)
            matcher_config = edict(matcher_config)

        with open(config['outlier_rejection_config'], 'r') as f_:
            outlier_rejection_config = yaml.load(f_, Loader=yaml.Loader)
            outlier_rejection_config = edict(outlier_rejection_config)
        config['kpfcn_config'] = matcher_config['kpfcn_config']

        # matcher initialization
        self.matcher = Matcher(matcher_config).to(device)  # pretrained point cloud matcher model
        state = torch.load(config.matcher_weights, weights_only=True)
        self.matcher.load_state_dict(state['state_dict'])
        print("Last epoch loaded for matcher:", state['epoch'])

        # outlier model initialization
        self.outlier_model = Outlier_Rejection(outlier_rejection_config.model).to(device)
        state = torch.load(config.outlier_rejection_weights, weights_only=True)
        self.outlier_model.load_state_dict(state['state_dict'])
        print("Last epoch loaded for outlier model:", state['epoch'])
        self.device = device

        self.kpfcn_config = config['kpfcn_config']
        # for debugging
        self.vis=False


    def inference(self, inputs, reject_outliers=True, inlier_thr=0.8, timer=None):

        self.matcher.eval()
        self.outlier_model.eval()
        with torch.no_grad():

            if timer: timer.tic("matcher")
            data = self.matcher(inputs, timers=None)
            if timer: timer.toc("matcher")

            if timer: timer.tic("outlier rejection")
            confidence = self.outlier_model(data)
            if timer: timer.toc("outlier rejection")

            inlier_conf = confidence[0]

            coarse_flow = data['coarse_flow'][0]
            inlier_mask, inlier_rate = NeCoLoss.compute_inlier_mask(data, inlier_thr, s2t_flow=coarse_flow)
            match_filtered = inlier_mask[0] [  inlier_conf > inlier_thr ]
            # plot
            if self.vis:
                plot(data, inlier_mask[0])
            
            inlier_rate_2 = match_filtered.sum()/(match_filtered.shape[0])
            vec_6d = data['vec_6d'][0]

            if reject_outliers:
                vec_6d = vec_6d [inlier_conf > inlier_thr]

            ldmk_s, ldmk_t = vec_6d[:, :3], vec_6d[:, 3:]


            return ldmk_s, ldmk_t, inlier_rate, inlier_rate_2
    

def plot(data, inlier_mask=None):
    from correspondence.lib.benchmark_utils import correspondence_viz_open3d
    import numpy as np
    
    s_pcd_raw = data ['src_pcd_list'][0]
    t_pcd_raw = data ['tgt_pcd_list'][0]
    s_pcd, t_pcd = data['s_pcd'][0], data['t_pcd'][0]
    s2t_flow = data['coarse_flow'][0]
    match_pred = data['coarse_match_pred']
    batched_rot = data['batched_rot']  # B,3,3
    batched_trn = data['batched_trn']

    # use the match prediction as the motion anchor
    match_pred_i = match_pred[ match_pred[:, 0] == 0 ]
    s_id , t_id = match_pred_i[:,1], match_pred_i[:,2]
    t_pcd_matched= t_pcd[t_id]
    
    # Transform source points: deform + rigid transform
    s_pcd_deformed = s_pcd + s2t_flow  # Apply deformation
    s_pcd_wrapped = (batched_rot[0] @ s_pcd_deformed.T + batched_trn[0]).T  # Apply rigid transform
    
    # Get transformed source points and target points
    s_pcd_matched_transformed = s_pcd_wrapped[s_id]  # In target frame!
    t_pcd_matched = t_pcd[t_id]  # Already in target frame

    # Compute inliers
    inlier_thr = 0.04
    distances_sq = torch.sum((s_pcd_matched_transformed - t_pcd_matched)**2, dim=1)
    inliers = distances_sq < (inlier_thr**2)

    s_id , t_id = match_pred[:,1], match_pred[:,2]
    corrs = np.stack([s_id.cpu().numpy(), t_id.cpu().numpy()], axis=0)

    correspondence_viz_open3d(s_pcd_raw, t_pcd_raw, s_pcd, t_pcd, corrs, inlier_mask=inliers)
    
    if inlier_mask is not None:
        # Convert to numpy if it's a tensor
        if torch.is_tensor(inlier_mask):
            inlier_mask = inlier_mask.cpu().numpy()
        
        n_corrs = corrs.shape[1]
        # filter correspondences and inliers with the mask
        corrs = corrs[:, inlier_mask]
        inliers = inliers[inlier_mask]

        # report how many correspondences were filtered
        print(f"{n_corrs - inlier_mask.sum()}/{n_corrs} correspondences filtered out.")

    correspondence_viz_open3d(s_pcd_raw, t_pcd_raw, s_pcd, t_pcd, corrs, inlier_mask=inliers)
