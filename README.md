# VI-Net

Code for "VI-Net: Boosting Category-level 6D Object Pose Estimation via Learning Decoupled Rotations on the Spherical Representations", ICCV 2023. [[Arxiv](https://arxiv.org/abs/2308.09916)]


Created by **[Jiehong Lin](https://jiehonglin.github.io/), Zewei Wei, [Yabin Zhang](https://ybzh.github.io/),  [Kui Jia](http://kuijia.site/)**.

![image](https://github.com/JiehongLin/VI-Net/blob/main/doc/overview.png)

## Requirements
The code has been tested with
- python 3.7.6
- pytorch 1.9.0
- CUDA 11.3

Other dependencies:

```
sh dependencies.sh
```

## Data Processing

Please refer to our another work of [Self-DPDN](https://github.com/JiehongLin/Self-DPDN).

## Model and Result Download

The trained models and test results are provided [here](https://drive.google.com/file/d/1t7Q6VxJXJFDh1nripFiLtaKPrrUgeYXR/view?usp=sharing).


## Network Training


Train VI-Net for rotation estimation:

```
python train.py --gpus 0 --dataset ${DATASET} --mode r
```

Train the network of [pointnet++](https://github.com/charlesq34/pointnet2) for translation and size estimation:

```
python train.py --gpus 0 --dataset ${DATASET} --mode ts 
```

The string "DATASET" could be set as `DATASET=REAL275` or `DATASET=CAMERA25`.

## Evaluation

To test the model, please run:

```
python test.py --gpus 0 --dataset ${DATASET}
```
The string "DATASET" could be set as `DATASET=REAL275` or `DATASET=CAMERA25`.

## Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{lin2023vi,
      title={Vi-net: Boosting category-level 6d object pose estimation via learning decoupled rotations on the spherical representations},
      author={Lin, Jiehong and Wei, Zewei and Zhang, Yabin and Jia, Kui},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={14001--14011},
      year={2023}
    }
    
## Acknowledgements

Our implementation leverages the code from [NOCS](https://github.com/hughw19/NOCS_CVPR2019), [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet), and [SPD](https://github.com/mentian/object-deformnet).

## License
Our code is released under MIT License (see LICENSE file for details).

## Contact
`mortimer.jh.lin@gmail.com`

