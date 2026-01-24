import os, sys, glob, torch
sys.path.append("../")
[sys.path.append(i) for i in ['.', '..']]
from pathlib import Path
import numpy as np
import torch
import random
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def find_new_corr(corr, mask_x, mask_y):
    old_to_new_x = {old_idx: new_idx for new_idx, old_idx in enumerate(np.flatnonzero(mask_x))}
    old_to_new_y = {old_idx: new_idx for new_idx, old_idx in enumerate(np.flatnonzero(mask_y))}

    # Step 2: Filter and remap correspondences
    new_corr = []
    for i, j in corr:
        if i in old_to_new_x and j in old_to_new_y:
            new_corr.append([old_to_new_x[i], old_to_new_y[j]])

    return np.array(new_corr, dtype=int)


class _Plants(Dataset):

    def __init__(self, config, split, data_augmentation=False, check_computed=None):
        super(_Plants, self).__init__()

        assert split in ['train','val','test']


        self.entries = self.read_entries(  config.split[split] , config.data_root, d_slice=None, check_computed=check_computed )

        self.base_dir = config.data_root
        self.data_augmentation = data_augmentation
        self.config = config

        self.rot_factor = 1.
        self.augment_noise = config.augment_noise
        self.max_points = 40_000  # 30000 TODO: or like in lepard 40_000 - disable downsampling for test though to get full res results at inference

        # self.overlap_radius = 0.0375  # does not seem to be needed for this dataset

        # self.cache = {}
        # self.cache_size = 30000



    def read_entries (self, split, data_root, d_slice=None, shuffle= False, check_computed=None):
        entries = glob.glob(os.path.join(data_root, split, "*/*.npz"), recursive=True)
        if check_computed is not None:
            entries = [e for e in entries if not (Path(check_computed) / (Path(e).stem + "_out.npz")).exists()]
        if shuffle:
            random.shuffle(entries)
        if d_slice:
            return entries[:d_slice]
        return entries


    def __len__(self):
        return len(self.entries )


    def __getitem__(self, index, debug=False):



        with np.load(self.entries[index]) as entry :

            # get transformation
            rot = entry['rot']
            trans = entry['trans']
            s2t_flow = entry['s2t_flow']
            src_pcd = entry['s_pc']
            tgt_pcd = entry['t_pc']
            correspondences = entry['correspondences']
            if "metric_index" in entry:
                metric_index = entry['metric_index'].squeeze()
            else:
                metric_index = None
            # Centering patch (global normalization)
            all_points = np.vstack([src_pcd, tgt_pcd])
            center = all_points.mean(axis=0, keepdims=True)
            src_pcd = src_pcd - center
            tgt_pcd = tgt_pcd - center

        depth_paths = None
        cam_intrin = None

        downsampled = False
        # if we get too many points, we do some downsampling
        #print(f"Number of source points: {src_pcd.shape[0]:_d}")
        if src_pcd.shape[0] > self.max_points:
            print("Downsampling...")
            downsampled = True
            pts_max = min(src_pcd.shape[0], tgt_pcd.shape[0])
            sub_idx_src = np.random.permutation(pts_max)[:self.max_points]
            src_pcd = src_pcd[sub_idx_src]
            s2t_flow = s2t_flow[sub_idx_src]
            # indices of target - no filtering
            sub_idx_tgt = np.arange(tgt_pcd.shape[0])
        # print(f"Number of target points: {tgt_pcd.shape[0]:_d}")
        if (tgt_pcd.shape[0] > self.max_points):
            print("Downsampling...")
            sub_idx_tgt = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[sub_idx_tgt]
            if not downsampled:
                sub_idx_src = np.arange(src_pcd.shape[0])
        
        src_pcd_deformed = src_pcd + s2t_flow
        if downsampled:
            correspondences = find_new_corr(correspondences, sub_idx_src, sub_idx_tgt)
            # assert that none of the important variables are empty 
            assert src_pcd.shape[0] > 0, "Source point cloud is empty after downsampling."
            assert tgt_pcd.shape[0] > 0, "Target point cloud is empty after downsampling."
            # assert correspondences.shape[0] > 0, "Correspondences are empty after downsampling."  # removed; empty corres may be good for the outlier rejection?
            assert s2t_flow.shape[0] > 0, "Scene flow is empty after downsampling."

        
        if debug:
            import open3d as o3d
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)

            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            src_wrapped_o3d = o3d.geometry.PointCloud()
            src_wrapped_o3d.points = o3d.utility.Vector3dVector(src_wrapped)
            src_wrapped_o3d.paint_uniform_color(c_pink)
            src_pcd_o3d = o3d.geometry.PointCloud()
            src_pcd_o3d.points = o3d.utility.Vector3dVector(src_pcd)
            src_pcd_o3d.paint_uniform_color(c_red)
            tgt_pcd_o3d = o3d.geometry.PointCloud()
            tgt_pcd_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
            tgt_pcd_o3d.paint_uniform_color(c_blue)
            o3d.visualization.draw_geometries([src_pcd_o3d, tgt_pcd_o3d, src_wrapped_o3d])


        # add gaussian noise
        if self.data_augmentation:
            print("Augmenting...")
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                src_pcd = np.matmul(rot_ab, src_pcd.T).T
                src_pcd_deformed = np.matmul(rot_ab, src_pcd_deformed.T).T
                rot = np.matmul(rot, rot_ab.T)
            else:
                tgt_pcd = np.matmul(rot_ab, tgt_pcd.T).T
                rot = np.matmul(rot_ab, rot)
                trans = np.matmul(rot_ab, trans)

            src_pcd += (np.random.rand(src_pcd.shape[0], 3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0], 3) - 0.5) * self.augment_noise
            s2t_flow = src_pcd_deformed - src_pcd


        if debug:
            src_wrapped = (np.matmul( rot, src_pcd_deformed.T ) + trans ).T
            src_wrapped_o3d = o3d.geometry.PointCloud()
            src_wrapped_o3d.points = o3d.utility.Vector3dVector(src_wrapped)
            src_wrapped_o3d.paint_uniform_color(c_red)
            tgt_pcd_o3d = o3d.geometry.PointCloud()
            tgt_pcd_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
            tgt_pcd_o3d.paint_uniform_color(c_blue)
            o3d.visualization.draw_geometries([tgt_pcd_o3d, src_wrapped_o3d])


        if (trans.ndim == 1):
            trans = trans[:, None]


        src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
        tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)


        #R * ( Ps + flow ) + t  = Pt
        return self.entries[index], src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trans, center, s2t_flow, metric_index, depth_paths, cam_intrin
