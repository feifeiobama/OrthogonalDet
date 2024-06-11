# OrthogonalDet
Code for CVPR 2024 paper [Exploring Orthogonality in Open World Object Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_Exploring_Orthogonality_in_Open_World_Object_Detection_CVPR_2024_paper.pdf).

<p align="center">
    <img src="assets/intro.png" alt="OrthogonalDet" width=60%>
</p>

## Requirements
- Linux or macOS with Python ≥ 3.8.
- Install [PyTorch ≥ 1.9.0, torchvision](https://pytorch.org/#install),
  [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html),
  timm, and einops.
- Prepare datasets:
  - Download [COCO](https://cocodataset.org/#download) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
  - Convert annotation format using `coco_to_voc.py`.
  - Move all images to `datasets/JPEGImages` and annotations to `datasets/Annotations`.

## Getting Started
* Training for open world object detection:
  ```
  bash run_owod.sh
  ```
  Evaluation for open world object detection:
  ```
  bash test_owod.sh
  ```
* Experiment for incremental object detection:
  ```
  bash run_iod.sh
  ```
* Visualize the results:
  ```
  python demo.py -i LIST_OF_IMAGES
  ```
* Note that we are using an ImageNet pre-trained backbone. To switch to a DINO pre-trained backbone, please download the [model weights](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) and then follow [these instructions](https://github.com/facebookresearch/detectron2/blob/main/tools/convert-torchvision-to-d2.py).

## Results
The following results were obtained with four NVIDIA 2080 Ti GPUs, using the checkpoints at [this link](https://drive.google.com/drive/folders/1XTTmjC_oVA2-J8l-bBdBo9nrQVq-lgKm).

* Open world object detection on M-OWODB and S-OWODB:
  
  ![owod](assets/owod.png)
  
* Incremental object detection on PASCAL VOC:
  
  ![iod](assets/iod.png)

## Citation

If you find this code useful, please consider citing:
```bibtex
@inproceedings{sun2024exploring,
  title={Exploring Orthogonality in Open World Object Detection},
  author={Sun, Zhicheng and Li, Jinghan and Mu, Yadong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17302--17312},
  year={2024},
}
```

## Acknowledgement

Our implementation is based on [RandBox](https://github.com/scuwyh2000/RandBox) which uses [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).
