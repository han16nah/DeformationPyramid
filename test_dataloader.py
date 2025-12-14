from model.geometry import *
import os
from pathlib import Path
import torch
import sys
sys.path.append("correspondence")


from tqdm import tqdm
import argparse



from model.registration import Registration
import  yaml
from easydict import EasyDict as edict
from model.loss import compute_flow_metrics

from utils.benchmark_utils import setup_seed
from utils.utils import Logger, AverageMeter
from utils.tiktok import Timers

from correspondence.landmark_estimator import Landmark_Model



def join(loader, node):
    seq = loader.construct_sequence(node)
    return '_'.join([str(i) for i in seq])
yaml.add_constructor('!join', join)

setup_seed(0)



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help= 'Path to the config file.')
    parser.add_argument('--visualize', action = 'store_true', help= 'visualize the registration results')
    parser.add_argument('--write', action = 'store_true', help= 'write the registration results to .npz')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    config['snapshot_dir'] = 'snapshot/%s/%s' % (config['folder'], config['exp_dir'])
    os.makedirs(config['snapshot_dir'], exist_ok=True)


    config = edict(config)


    # backup the experiment
    #os.system(f'cp -r config {config.snapshot_dir}')
    #os.system(f'cp -r data {config.snapshot_dir}')
    #os.system(f'cp -r model {config.snapshot_dir}')
    #os.system(f'cp -r utils {config.snapshot_dir}')


    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    ldmk_model =  Landmark_Model(config_file = config.ldmk_config, device=config.device)
    config['kpfcn_config'] = ldmk_model.kpfcn_config

    model = Registration(config)
    timer = Timers()


    from correspondence.datasets._plants import _Plants
    from correspondence.datasets.dataloader import get_dataloader


    # splits = [ '4DMatch-F', '4DLoMatch-F' ]
    splits = ['train', 'val']


    for split in splits:

        stats_meter = None
        set = _Plants(config, split, data_augmentation=False)
        loader, _ = get_dataloader(set, config, shuffle=False)


        logger = Logger(os.path.join(config.snapshot_dir, config.split["test"] + ".log"))

        num_iter =  len(set)
        c_loader_iter = loader.__iter__()

        for c_iter in tqdm(range(num_iter)):

            inputs = next(c_loader_iter)


            for k, v in inputs.items():
                if type(v) == list:
                    inputs [k] = [item.to(config.device) for item in v if type(item) != str]
                    if k == "entry_list":
                        entry_list = v[0]
                elif type(v) in [dict, float, type(None), np.ndarray]:
                    pass
                else:
                    inputs [k] = v.to(config.device)
            
            src_pcd, tgt_pcd = inputs["src_pcd_list"][0], inputs["tgt_pcd_list"][0]
            s2t_flow = inputs['sflow_list'][0]
            rot, trn = inputs['batched_rot'][0],  inputs['batched_trn'][0]
            correspondence = inputs['correspondences_list'][0]


            """compute scene flow GT"""
            src_pcd_deformed = src_pcd + s2t_flow
            s_pc_wrapped = ( rot @ src_pcd_deformed.T + trn ).T
            s2t_flow = s_pc_wrapped - src_pcd
            flow_gt = s2t_flow.to(config.device)


            """compute overlap mask"""
            overlap = torch.zeros(len(src_pcd))
            try:
                overlap[correspondence[:, 0].long()] = 1
            except IndexError as e:
                # this happens in a subset where there are no correspondences
                print(f"IndexError for {entry_list} when creating overlap mask: {e}. Setting overlap to all zeros.")
                overlap = torch.zeros(len(src_pcd))
            overlap = overlap.bool()
            overlap =  overlap.to(config.device)