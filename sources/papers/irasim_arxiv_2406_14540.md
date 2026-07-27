# IRASim: A Fine-Grained World Model for Robot Manipulation（arXiv:2406.14540）

> 来源归档（ingest）

- **标题：** IRASim: A Fine-Grained World Model for Robot Manipulation
- **类型：** paper / world model / trajectory-to-video / policy evaluation / model-based planning
- **arXiv：** <https://arxiv.org/abs/2406.14540>（PDF：<https://arxiv.org/pdf/2406.14540.pdf>）
- **项目页：** <https://gen-irasim.github.io/>
- **代码：** <https://github.com/bytedance/IRASim>（Apache-2.0）
- **数据 / 权重：** ByteDance CDN + Hugging Face [`fangqi/IRASim`](https://huggingface.co/datasets/fangqi/IRASim)
- **作者：** Fangqi Zhu、Hongtao Wu†∗、Song Guo∗、Yuxiao Liu、Chilam Cheang、Tao Kong（† Project Lead；∗ Corresponding）
- **机构：** 香港科技大学（HKUST）、字节跳动 Seed（ByteDance Seed）
- **入库日期：** 2026-07-27
- **一句话说明：** 用 **Diffusion Transformer + 帧级动作条件（Frame-Ada）** 做 trajectory-to-video：给定历史观测与动作轨迹，生成细粒度机–物交互视频；支撑策略评估（与 LIBERO 仿真相关高）、模型规划（Push-T IoU **0.637→0.961**）与键盘/VR 可控合成。

## 开源状态（项目页 + 仓库核查，2026-07-27）

- **已开源：** [bytedance/IRASim](https://github.com/bytedance/IRASim) · **Apache-2.0**；含训练 / 评估脚本、`application/languagetable.py` 交互 demo；RT-1 / Bridge / Language-Table **训练数据、评测数据与 checkpoints** 可经 `scripts/download.sh` 或 HF 拉取。
- 项目页：<https://gen-irasim.github.io/> 展示 demo 与论文入口。

## 摘要级要点

- **任务：** trajectory-to-video — \(I^{t+1:t+n+1}=f(I^{t-h:t},a^{t:t+n})\)；动作块级条件（非仅文本）。
- **关键设计：** Video-Ada（整段轨迹单 embedding）vs **Frame-Ada**（每帧对应动作 embedding + AdaLN）；空间–时间注意力；SDXL VAE latent 扩散。
- **数据：** RT-1、Bridge、Language-Table、RoboNet；分辨率最高约 **288×512**，长 horizon 可 **150+** 帧自回归。
- **下游：** 策略评估与 GT 仿真相关；模型规划显著抬升 Push-T；键盘/VR 控制数据集中虚拟臂。

## 核心论文摘录（MVP）

### 1) Frame-level action conditioning

- **链接：** §3.3；Fig. 2
- **摘录要点：** 轨迹相对文本是更细粒度条件；Frame-Ada 为每帧编码动作并注入空间块 scale/shift；历史帧保持干净、仅对预测帧加噪。
- **对 wiki 的映射：**
  - [IRASim](../../wiki/entities/paper-irasim.md) — 核心接口。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 像素/ latent 动作条件谱系。

### 2) 策略评估与规划

- **链接：** §4；Push-T IoU 0.637→0.961
- **摘录要点：** IRASim 评测与 LIBERO GT 强相关；测试时用模型规划筛轨迹提案；算力增加可抬升规划收益。
- **对 wiki 的映射：**
  - [world-model-physics-fidelity-outputs](../../wiki/overview/world-model-physics-fidelity-outputs.md) — 未来视频输出族。
  - [Masked Visual Actions](../../wiki/entities/paper-masked-visual-actions.md) — 同属视频沙盒评估轴。

### 3) 开源可复现

- **链接：** 项目页；README Installation / Dataset / Training
- **摘录要点：** `scripts/install.sh`、`download.sh`、`main.py --config configs/train/...`、`application/languagetable.py`。
- **对 wiki 的映射：**
  - [`sources/repos/irasim.md`](../repos/irasim.md)

## BibTeX

```bibtex
@article{zhu2024irasim,
  title   = {IRASim: A Fine-Grained World Model for Robot Manipulation},
  author  = {Zhu, Fangqi and Wu, Hongtao and Guo, Song and Liu, Yuxiao and Cheang, Chilam and Kong, Tao},
  journal = {arXiv preprint arXiv:2406.14540},
  year    = {2024}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-irasim.md`](../../wiki/entities/paper-irasim.md)
- 项目页：[`sources/sites/gen-irasim-github-io.md`](../sites/gen-irasim-github-io.md)
- 代码：[`sources/repos/irasim.md`](../repos/irasim.md)
- 策展语境：[`sources/blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md`](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
