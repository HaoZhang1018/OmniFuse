# [IEEE TPAMI 2025] OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics.
This repository is the official implementation of the **IEEE TPAMI 2025** paper:
_"OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics"_ 
### [Paper](https://ieeexplore.ieee.org/abstract/document/10994384) | [Code](https://github.com/HaoZhang1018/OmniFuse) 
## Functions
![Schematic diagram of OmniFuse's functions.](https://github.com/HaoZhang1018/OmniFuse/blob/main/Display/Functions.png)

## Environmental Installation
```
conda create -n OmniFuse python=3.9.19
conda activate OmniFuse
```
It is recommended to use the following versions of the Torch architecture.
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
```
pip install -U openmim
mim install mmcv-full==1.7.2
```
```
cd segment-anything
pip install .
cd ..
```
Installing additional dependencies
```
pip install -r requirements.txt
```


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{zhang2025omnifuse,
  title={OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics},
  author={Zhang, Hao and Cao, Lei and Zuo, Xuhui and Shao, Zhenfeng and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```
## Illustrate
- [ ] OmniFuse is highly robust to the real scenario where both multi-source images exhibit composite degradation. Typical degradation types are considered, including inadequate illumination, noise, and color cast in visible images, and low contrast, noise, and uneven stripes in infrared images.
- [ ] OmniFuse supports language instructions to achieve fine-grained control over the fusion process, emphasizing semantic objects of interest, which potentially facilitates downstream semantic applications.
      
