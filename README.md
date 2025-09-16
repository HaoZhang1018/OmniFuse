# [IEEE TPAMI 2025] OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics.
This repository is the official implementation of the **IEEE TPAMI 2025** paper:
_"OmniFuse: Composite Degradation-Robust Image Fusion with Language-Driven Semantics"_ 
### [Paper](https://ieeexplore.ieee.org/abstract/document/10994384) | [Code](https://github.com/HaoZhang1018/OmniFuse) 
## Functions
![Schematic diagram of OmniFuse's functions.](https://github.com/HaoZhang1018/OmniFuse/blob/main/Display/Functions.png)

## ⚙️ Environmental Installation
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
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
****
### Preparing tools
The model requires pre-trained weights for GroundingDINO-SAM and BERT.
Navigate to the following directories [\[Instruction Path\]](https://github.com/HaoZhang1018/OmniFuse/tree/main/models/bert-base-uncased)  [\[Instruction Path\]](https://github.com/HaoZhang1018/OmniFuse/tree/main/models/groundingdino_sam_checkpoints) and download the pre-trained parameters as instructed.
Place the downloaded weights in the specified paths accordingly.
```
-  BERT: Place the checkpoint in : './models/bert-base-uncased/'
- GroundingDINO-SAM: Place the model in: './models/groundingdino_sam_checkpoints/'
```
## ❄️ Test
### Prepare Dataset
Please place the data in the following path (Note: X indicates visible images, and Y indicates infrared images):
```
./datasets/
```
### pre-trained weights
We provide pre-trained model parameters. Please download them according to the instruction file paths [\[Instruction Path\]](https://github.com/HaoZhang1018/OmniFuse/tree/main/pretrained) at the following addresses, and save these pre-trained models under the following folder path.
```
./pretrained/
```
### Run 
You can modify the parameter settings in test_Fusion.yaml, such as setting **Fusion_Model_type** to **base** or **modulated**, depending on whether text modulation is used. Then run the code.
```
python test_Fusion.py -opt ./options/test/test_Fusion.yml
```

## 🔥 Train
### Prepare Dataset
Please place the data in the following path (Note: X indicates visible images, and Y indicates infrared images):
```
./datasets/
```
### train AutoEncoder model 
```
python train_AE.py -opt ./options/train/train_AE.yml
```
### train LatentDiffusion model 
```
python train_latentdiffusion.py -opt ./options/train/train_LatentDif_VIS.yml
python train_latentdiffusion.py -opt ./options/train/train_LatentDif_IR.yml
```
### train Fusion model
The training of the fusion model is divided into two parts. The first is the training of the base model, which is done by setting the training parameter **Fusion_Model_type** to **base**. The pretrained parameters, basic fusion results, and segmentation parameters obtained from this process will serve as the foundation for training in the **modulated** mode.
```
python train_Fusion.py -opt ./options/train/train_Fusion.yml
```

## 📕 Citation
If our work assists your research, feel free to give us a star or cite us using:
```
@article{zhang2025omnifuse,
  title={OmniFuse: Composite degradation-robust image fusion with language-driven semantics},
  author={Zhang, Hao and Cao, Lei and Zuo, Xuhui and Shao, Zhenfeng and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={9},
  pages={7577--7595},
  year={2025},
  publisher={IEEE}
}
```
## 📢 Illustrate
- [ ] OmniFuse is highly robust to the real scenario where both multi-source images exhibit composite degradation. Typical degradation types are considered, including inadequate illumination, noise, and color cast in visible images, and low contrast, noise, and uneven stripes in infrared images.
- [ ] OmniFuse supports language instructions to achieve fine-grained control over the fusion process, emphasizing semantic objects of interest, which potentially facilitates downstream semantic applications.
## 🙏 Acknowledgements
Our code is built upon the following libraries. We sincerely thank the authors for their contributions. If you use any components or pretrained weights from these works, please make sure to cite the corresponding references.
#### [IR-SDR](https://github.com/Algolzw/image-restoration-sde) | [SegNeXt](https://github.com/Visual-Attention-Network/SegNeXt) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | [SAM](https://github.com/facebookresearch/segment-anything) 
