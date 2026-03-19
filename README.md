<p align="center">

  <h1 align="center">DROID-W: DROID-SLAM in the Wild</h1>
  <p align="center">
    <a href="https://moyangli00.github.io/"><strong>Moyang Li*</strong></a>
    .
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu*</strong></a>
    .
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
    .
    <a href="https://cvg.ethz.ch/team/Dr-Daniel-Bela-Barath"><strong>Dániel Béla Baráth</strong></a>
</p>
<p align="center"> <strong>Computer Vision And Pattern Recognition (CVPR) 2026</strong></p>
  <h3 align="center"><a href="https://moyangli00.github.io/droid-w/static/docs/droid_w_paper.pdf">Paper</a> | <a href="https://moyangli00.github.io/droid-w/">Project Page</a> | <a href="https://cvg-data.inf.ethz.ch/DROID-W">Dataset</a></h3>
  <div align="center"></div>
</p>
<p align="center">
    <img src="./media/teaser.png" alt="teaser_image" width="100%">
</p>

<p align="center">
Given a casually captured in-the-wild video, DROID-W estimates accurate camera trajectory, scene structure and dynamic uncertainty.
</p>
<br>

<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Installation

1. First you have to make sure that you clone the repo with the `--recursive` flag.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 
```bash
git clone --recursive https://github.com/MoyangLi00/DROID-W.git
cd DROID-W
```

2. Creating a new conda environment. 
```bash
conda create --name droid-w python=3.10
conda activate droid-w
```

3. Install CUDA 11.8 and torch-related pacakges
```bash
pip install numpy==1.26.3
conda install --channel "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip3 install -U xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118
```

4. Install the remaining dependencies.
```bash
python -m pip install -e thirdparty/lietorch --no-build-isolation
python -m pip install -e thirdparty/diff-gaussian-rasterization-w-pose --no-build-isolation
python -m pip install -e thirdparty/simple-knn --no-build-isolation
```

5. Check installation.
```bash
python -c "import torch; import lietorch; import simple_knn; import diff_gaussian_rasterization; print(torch.cuda.is_available())"
```
6. Now install the droid backends and the other requirements
```bash
python -m pip install -e . --no-build-isolation
python -m pip install -r requirements.txt
```
7. Install MMCV (used by metric depth estimator)
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```
8. Download the pretained models [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing), put it inside the `pretrained` folder.

## Run

### Bonn Dynamic Dataset
Download the data as below and the data is saved into the `./Datasets/Bonn` folder. Note that the script only downloads the 8 sequences reported in the paper. To get other sequences, you can download from the [webiste of Bonn Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html).
```bash
bash scripts_downloading/download_bonn.sh
```
You can run DROID-W via the following command:
```bash
python run.py  ./configs/Dynamic/Bonn/{config_file}
```
We have prepared config files for the 8 sequences. Note that this dataset needs preprocessing the pose. We have implemented that in the dataloader. If you want to test with sequences other than the ones provided, don't forget to specify ```dataset: 'bonn_dynamic'``` in your config file. The easiest way is to inherit from ```bonn_dynamic.yaml```.

### TUM RGB-D (dynamic) Dataset
Download the data (9 dynamic sequences) as below and the data is saved into the `./Datasets/TUM_RGBD` folder. 
```bash
bash scripts_downloading/download_tum.sh
```
The config files for 9 dynamic sequences of this dataset can be found under ```./configs/Dynamic/TUM_RGBD```. You can run DROID as the following:
```bash
python run.py --config ./configs/Dynamic/Wild_SLAM_Mocap/{config_file} 
```

### DyCheck Dataset
Download the [DyCheck](https://drive.google.com/drive/folders/1BHzjHo58nGAMvKMo_AS0_SwU2tJagXXx) dataset and put it in the `./datasets/DyCheck` folder.
The config files for the 12 sequences of this dataset can be found under ```./configs/Dynamic/DyCheck```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/DyCheck/{config_file}
```

### DROID-W Dataset
Download the data (7 dynamic sequences) as below and the data is saved into the `./Datasets/DROID-W` folder. 
```bash
bash scripts_downloading/download_droidw.sh
```
  The config files for 7 dynamic sequences of this dataset can be found under ```./configs/Dynamic/DROIDW```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/DROIDW/{config_file} 
```

### YouTube Sequences
Download the sequences (12 dynamic sequences) as below and the data is saved into the `./Datasets/YouTube` folder.
```bash
bash scripts_downloading/download_youtube.sh
```
  The config files for 12 dynamic sequences of this dataset can be found under ```./configs/Dynamic/YouTube```. You can run DROID-W as the following:
```bash
python run.py --config ./configs/Dynamic/YouTube/{config_file} 
```


## Evaluation

### Camera poses
The camera trajectories will be automatically evaluated after each run of DROID-W except for the DROID-W dataset. Evaluate the camera poses for the DROID-W dataset using the following command:
```bash
python scripts_eval/evaluate_droidw.py
```

We provide a python script to summarize the RMSE of ATE:
```bash
python scripts_eval/summarize_rmse.py -b {path_to_output_dir}
```

## Note

We disable the Gaussian Splatting mapping module by default. If you want to enable it, please set `mapping.enable: True` in `configs/droid_w.yaml`.

## Acknowledgement
We adapted some codes from some awesome repositories including [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [WildGS-SLAM](https://github.com/GradientSpaces/WildGS-SLAM), and [Metric3D V2](https://github.com/YvanYin/Metric3D). Thanks for making codes publicly available. 

## Citation

If you find our code or paper useful, please cite
```bibtex
@inproceedings{Li2026DROIDW,
  author    = {Li, Moyang and Zhu, Zihan and Pollefeys, Marc and Barath, Daniel},
  title     = {DROID-SLAM in the Wild},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Contact
Contact [Moyang Li](mailto:limoyang2000@gmail.com) for questions, comments and reporting bugs.
