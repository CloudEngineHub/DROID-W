import numpy as np
import os
import sys
import glob
import json
import shutil
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.spatial.transform import Rotation
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
from evo.core import metrics
from evo.tools import plot
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def align_traj(ref_traj_path, est_traj_path, plot_name="traj_error", plot_parent_dir="plots"):
    if not os.path.exists(ref_traj_path):
        print(f"Reference trajectory file not found: {ref_traj_path}")
        return
    if not os.path.exists(est_traj_path):
        print(f"Estimated trajectory file not found: {est_traj_path}")
        return
    ref_file = np.loadtxt(ref_traj_path)
    est_file = np.loadtxt(est_traj_path)

    traj_SE3_est = est_file[:, 1:8]
    traj_SE3_ref = ref_file[:, 1:8]
    
    timestamps_est = est_file[:, 0]
    timestamps_ref = ref_file[:, 0]

    traj_xyz_est = traj_SE3_est[:, 0:3]
    traj_xyz_ref = traj_SE3_ref[:, 0:3]
    traj_quat_est = traj_SE3_est[:, 3:7]
    traj_quat_ref = traj_SE3_ref[:, 3:7]

    # xyzw to wxzy
    traj_quat_est = np.roll(traj_quat_est, 1, axis=1)
    traj_quat_ref = np.roll(traj_quat_ref, 1, axis=1)

    traj_xyz_est = [traj_xyz_est[i] for i in range(traj_xyz_est.shape[0])]
    traj_xyz_ref = [traj_xyz_ref[i] for i in range(traj_xyz_ref.shape[0])]
    traj_quat_est = [traj_quat_est[i] for i in range(traj_quat_est.shape[0])]
    traj_quat_ref = [traj_quat_ref[i] for i in range(traj_quat_ref.shape[0])]

    print("Aligning trajectories...")
    traj_est = PoseTrajectory3D(positions_xyz=traj_xyz_est, orientations_quat_wxyz=traj_quat_est, timestamps=timestamps_est)
    traj_ref = PoseTrajectory3D(positions_xyz=traj_xyz_ref, orientations_quat_wxyz=traj_quat_ref, timestamps=timestamps_ref)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff=0.1)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    print("Calculating APE ...")
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    print("Plotting ...")

    plot_collection = plot.PlotCollection("kf factor graph")
    fig_1 = plt.figure(figsize=(10, 7))
    plot_mode = plot.PlotMode.xyz
    ax = plot.prepare_axis(fig_1, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(
    ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], 
    title="Downtown " + plot_name[-1])
    # title="APE mapped onto trajectory, RMSE: %.2f m" % (ape_statistics["rmse"]))
    # set everything to bold and all use serif font
    plt.rcParams.update({'font.family': 'serif'})
    plt.rcParams.update({'font.weight': 200})
    ax.set_title(f"Downtown {int(plot_name[-1])}", fontsize=30)
    ax.set_xlabel("X (m)", fontsize=20, labelpad=20)
    ax.set_ylabel("Y (m)", fontsize=20, labelpad=20)
    ax.set_zlabel("Z (m)", fontsize=20, labelpad=25)
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='y', pad=8)
    ax.tick_params(axis='z', pad=10)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    ax.grid(True)
    ax.legend(fontsize=20)
    # set the size of colobar font to 20
    cbar = ax.figure.axes[-1]
    cbar.tick_params(labelsize=19)
    plot_collection.add_figure("2d", fig_1)
    plot_collection.export(f"{plot_parent_dir}/{plot_name}_3d.png", False)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    print(output_str)
    print("#"*34)
    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)

    return ape_statistics

def main(base_dir: str):
    gt_dir = "./datasets/DROID-W"
    scene_list = sorted(os.listdir(base_dir))
    print(f'Evaluating {len(scene_list)} scenes...')
    for scene_name in scene_list:
        if os.path.isdir(os.path.join(base_dir, scene_name)):
            print(f'Evaluating {scene_name}...')
            if os.path.exists(f'{gt_dir}/{scene_name}/traj_gt.txt'):
                gt_seq_txt = f'{gt_dir}/{scene_name}/traj_gt.txt'
            else:
                gt_seq_txt = f'{gt_dir}/{scene_name}/traj_gt_fastlivo.txt'
            save_path = f'{base_dir}/{scene_name}/traj'
            estimated_poses_file = f'{save_path}/est_poses_full.txt'
            timestamps_file = f'{base_dir}/{scene_name}/timestamps.txt'
            if not os.path.exists(estimated_poses_file):
                print(f"Estimated poses file not found: {estimated_poses_file}")
                continue
            if not os.path.exists(timestamps_file):
                print(f"Timestamps file not found: {timestamps_file}")
                continue
            # load estimated poses and timestamps
            estimated_poses = np.loadtxt(estimated_poses_file)
            timestamps = np.loadtxt(timestamps_file)
            assert len(estimated_poses) <= len(timestamps)
            estimated_poses[:, 0] = timestamps[:len(estimated_poses)]
            # save new estimated poses to a txt file
            outputfile = f'{save_path}/est_poses_full_new.txt'
            np.savetxt(outputfile, estimated_poses)
            os.makedirs(save_path, exist_ok=True)
            # remove all png and zip files in the save_path
            for file in os.listdir(save_path):
                if file.endswith('.png') or file.endswith('.zip'):
                    os.remove(os.path.join(save_path, file))
                # remove the scene_name folder
                if os.path.exists(os.path.join(save_path, scene_name)):
                    shutil.rmtree(os.path.join(save_path, scene_name))
            align_traj(gt_seq_txt, outputfile, "traj_" + scene_name, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DROID-W trajectories.")
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default="./Outputs/DROID-W",
        help="Base directory containing per-scene outputs.",
    )
    args = parser.parse_args()
    main(args.base_dir)