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
    os.system(f'cp -r config {config.snapshot_dir}')
    os.system(f'cp -r data {config.snapshot_dir}')
    os.system(f'cp -r model {config.snapshot_dir}')
    os.system(f'cp -r utils {config.snapshot_dir}')


    if config.gpu_mode:
        config.device = torch.cuda.current_device()
    else:
        config.device = torch.device('cpu')


    ldmk_model =  Landmark_Model(config_file = config.ldmk_config, device=config.device)
    config['kpfcn_config'] = ldmk_model.kpfcn_config

    model = Registration(config)
    timer = Timers()




    from correspondence.datasets._4dmatch import _4DMatch
    from correspondence.datasets._plants import _Plants
    from correspondence.datasets.dataloader import get_dataloader


    # splits = [ '4DMatch-F', '4DLoMatch-F' ]
    splits = ['test']


    for split in splits:

        #config.split['test'] = split

        stats_meter = None
        test_set = _Plants(config, 'test', data_augmentation=False, check_computed=config['snapshot_dir'] if args.write else None)
        test_loader, _ = get_dataloader(test_set, config, shuffle=False)


        logger = Logger(os.path.join(config.snapshot_dir, config.split["test"] + ".log"))

        num_iter =  len(test_set)
        c_loader_iter = test_loader.__iter__()

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


            """predict landmarks"""
            try:
                ldmk_s, ldmk_t, inlier_rate, inlier_rate_2 = ldmk_model.inference (inputs, reject_outliers=config.reject_outliers, inlier_thr=config.inlier_thr, timer=timer)
            except IndexError as e:
                print(f"IndexError for {entry_list} during landmark inference: {e}. Skipping this entry.")
                print("Corresponding inputs:", inputs)
                continue

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
                print(f"IndexError for {entry_list} when creating overlap mask: {e}. Setting overlap to all zeros.")
                overlap = torch.zeros(len(src_pcd))
            overlap = overlap.bool()
            overlap =  overlap.to(config.device)



            if config.deformation_model in ["NDP"]:
                model.load_pcds(src_pcd, tgt_pcd, landmarks=(ldmk_s, ldmk_t))

                timer.tic("registration")
                warped_pcd, iter, timer = model.register(visualize=args.visualize, timer = timer)
                timer.toc("registration")
                flow = warped_pcd - model.src_pcd

                for key, value in iter.items():
                    timer.tictoc(key, value)

                if args.write:
                    fstem = Path(entry_list).stem
                    print(f"Saving file to {Path(config['snapshot_dir']) / f'{fstem}_out.npz'}")
                    # de-center
                    center = inputs['center_list'][0]
                    src_pcd = src_pcd.cpu().numpy() + center.cpu().numpy()
                    tgt_pcd = tgt_pcd.cpu().numpy() + center.cpu().numpy()
                    warped_pcd = warped_pcd.cpu().numpy() + center.cpu().numpy()
                    # save data to .npz
                    np.savez(Path(config['snapshot_dir']) / f'{fstem}_out.npz',
                            s_pc=src_pcd,
                            t_pc=tgt_pcd,
                            s2t_flow=flow.cpu().numpy(),
                            s2t_flow_gt=flow_gt.cpu().numpy(),
                            warped_pcd=warped_pcd)


            elif config.deformation_model == "ED": # Lepard+NICP

                model.load_pcds(src_pcd, tgt_pcd)

                depth_paths = inputs['depth_paths_list'][0]
                cam_intrin = inputs['cam_intrin']

                # get pixel landmarks
                uv_src = xyz_2_uv(ldmk_s, cam_intrin)
                uv_tgt = xyz_2_uv(ldmk_t, cam_intrin)
                landmarks = (uv_src.to(config.device), uv_tgt.to(config.device))


                timer.tic("graph construction")
                model.load_raw_pcds_from_depth(depth_paths[0], depth_paths[1], cam_intrin, landmarks=landmarks)
                timer.toc("graph construction")


                timer.tic("registration")
                warped_pcd, point_mask = model.register(visualize=args.visualize)
                timer.toc("registration")

                flow = warped_pcd - model.src_pcd[point_mask]
                flow_gt = flow_gt[point_mask]
                overlap = overlap[point_mask]


            else:
                raise KeyError()



            metric_info = compute_flow_metrics(flow, flow_gt, overlap=overlap)


            if stats_meter is None:
                stats_meter = dict()
                for key, _ in metric_info.items():
                    stats_meter[key] = AverageMeter()
            for key, value in metric_info.items():
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                stats_meter[key].update(value)




        message = f'{c_iter}/{len(test_set)}: '
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.3f}\t'
        logger.write(message + '\n')

        print("score on ", split, '\n', message)




    # note down average time cost
    print('time cost average')
    for ele in timer.get_strings():
        logger.write(ele + '\n')
        print(ele)
