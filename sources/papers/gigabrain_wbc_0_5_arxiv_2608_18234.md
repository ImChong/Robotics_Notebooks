# GigaBrain-WBC-0.5（arXiv:2608.18234）

> 来源归档（ingest）

- **标题：** GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction
- **类型：** paper / humanoid / whole-body-tracking / behavior-world-model / terrain-interaction / unitree-g1
- **arXiv abs：** <https://arxiv.org/abs/2608.18234>
- **PDF：** <https://arxiv.org/pdf/2608.18234>
- **HTML：** <https://arxiv.org/html/2608.18234>
- **项目页：** <https://shepherd1226.github.io/gigabrain-wbc-0.5/>
- **机构：** 清华大学（Tsinghua University）；极佳视界（GigaAI）；上海理工大学（University of Shanghai for Science and Technology）；北京交通大学（Beijing Jiaotong University）；中国科学院自动化研究所（Institute of Automation, CAS）；中国科学院大学（University of Chinese Academy of Sciences）
- **作者：** Ziyang Cheng、Tianshu Tang、Jinxin Lan、Xinze Chen、Yuhan Gong、Zhichao Liu、Changzhong Wu、Yahao Mao、Zongyan Deng、Mingxuan Ma、Huasen Xi、Yilong Liu、Yutong Wu、Xiaofeng Wang、Yang Wang、Yun Ye、Guan Huang、Xiaojie Jin、Zheng Zhu#、Jiwen Lu#
- **发表 / 上传：** 2026-08-18（arXiv v1）；**修订：** 2026-11-05（[v5](https://arxiv.org/abs/2608.18234v5)）
- **代码：** <https://github.com/open-gigaai/giga-brain-wbc>（已开源，530 文件；见 [`sources/repos/giga-brain-wbc.md`](../repos/giga-brain-wbc.md)）
- **训练栈：** Isaac Lab + PPO；Unitree G1（29 DoF @ 50 Hz）+ Maker L01
- **入库日期：** 2026-08-21
- **通读：** 已完成
- **核对范围：** 2026-11-21 通读 `arXiv:2608.18234v5` 全文 PDF（978 行抽取文本：§1–§5、Table 1–5 与附录）及官方代码仓 `open-gigaai/giga-brain-wbc` README/训练与评测入口（`giga_brain_wbc/train.py`、`ppo_trainer_aux_loss.py`、`export.py`、`evaluation/run.py`）。
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.18234](https://arxiv.org/abs/2608.18234) | 论文与附录 |
| 项目页 | [gigabrain-wbc-0.5](https://shepherd1226.github.io/gigabrain-wbc-0.5/) | 真机对比视频、能力矩阵 |
| 数据 | Bones-Seed / MotionMillion / MotionDecode | 识别 terrain-interaction 子集混合 flat-ground |
| 基线 | SONIC / HoloMotion-1 / Humanoid-GPT | Table 3 sim-to-sim 对照 |

## 开源状态（步骤 2.5，2026-11-21 复核）

- **已开源：** 官方已发布 GitHub 仓库 [open-gigaai/giga-brain-wbc](https://github.com/open-gigaai/giga-brain-wbc)（Apache-2.0，530 个文件，含 Unitree G1 29-DoF 与 Maker L01 Isaac Lab 训练、ONNX 编码器/解码器导出、MuJoCo 基准评测与真机/PICO 遥操作部署）与 Hugging Face 数据集 [`open-gigaai/GigaBrain-WBC-0.5-TerrainData`](https://huggingface.co/datasets/open-gigaai/GigaBrain-WBC-0.5-TerrainData)（459 MB `bundles/data.tar`）。
- **处理：** 归档 [`sources/repos/giga-brain-wbc.md`](../repos/giga-brain-wbc.md)，并在 wiki 实体页补充真实入口符号与 `sequenceDiagram`。

## 摘要级要点

- **问题：** 现有 whole-body motion tracker 多在空场景平地训练，不学 terrain/object 接触如何改变动力学；靠扩大 reference corpus 换 OOD 鲁棒在环境相关可行集上失效。
- **主张：** 首个 humanoid **Behavior World Model（BWM）** — causal Transformer 每步联合输出 action、next proprio state、next latent command **GMM**。
- **Terrain pipeline：** 从 retarget 运动自动恢复 **全 3D** 接触几何（chairs、tables、stairs、boxes），规模可达现有 motion dataset 量级。
- **部署 filter：** 用上一步预测的 GMM 对当前 raw command 做 Mahalanobis test；OOD 时沿 ray radial retract 到 safety ellipsoid（非 replace by mean）。
- **训练：** 混合 flat + terrain + fallen initialization；fall recovery 训进 tracker 本体（非独立 get-up controller）。
- **结果（MuJoCo sim-to-sim）：** Terrain SR **81.3%**（SONIC 15.3%）；OOD SR **83.1%**；Fall SR **99.3%**（SONIC 5.9%）；Standard tracking MPKPE **76.6** mm（SONIC 82.3）。
- **硬件：** G1 真机物体/地形交互；G1 checkpoint fine-tune 迁 Maker L01。

## 核心摘录（面向 wiki 编译）

### 1) 架构（§3）

- Reference window \(c_t\)：10 帧 ×（29 joint pos + root rel rot/trans + gravity in ref frame）→ MLP encoder → latent \(z_t^{raw}\)（64-d，两 32-d token）。
- 训练期 FSQ + decoder 做 cycle-consistency（policy 路径用 unquantized latent）。
- 6-layer causal Transformer + KV cache；三头：action \(a_t\)、next state \(s_{t+1}\)、next-command GMM \(G_{t+1}\)。
- 部署：Mahalanobis \(M^2_{k^*}(z_t^{raw}) > R_{safe}^2\) 时 radial retract；\(R_{safe}\) 运行时可调。

### 2) 训练数据（§3.5 / Fig. 4）

| Corpus | 总量 | 识别 terrain-interaction |
|--------|------|--------------------------|
| Bones-Seed | 288 h | 12.50 h |
| MotionMillion | 900 h | 22.22 h |
| MotionDecode | 1000 h | 37.85 h |

- 与 flat-ground motion  controlled mix；terrain 几何 per-env spawn；fallen pose curriculum。

### 3) Table 3 四 regime（部署 \(R_{safe}=3\)）

| Method | Std SR↑ | Terrain SR↑ | OOD SR↑ | Fall SR↑† |
|--------|---------|-------------|---------|-----------|
| SONIC | 94.1 | 15.3 | 50.0 | 5.9 |
| HoloMotion-1 | 89.0 | 18.7 | 67.7 | 0.7 |
| Humanoid-GPT | 91.9 | 14.0 | 70.6 | 2.9 |
| **Ours** | **96.3** | **81.3** | **83.1** | **99.3** |

†Fall SR 为 recovery rate，与其他 SR 定义不同。

### 4) 能力矩阵（Table 1 / 项目页）

相对 GMT/TWIST/SONIC/HoloMotion-1/Humanoid-GPT/SceneBot/CMP/BFM-Zero，本文宣称唯一 ✓ 覆盖 teleop + terrain + object + OOD + fall robust。

## 对 wiki 的映射

- 沉淀实体页：[GigaBrain-WBC-0.5](../../wiki/entities/paper-gigabrain-wbc-0-5.md)
- 交叉补强：[SONIC 方法](../../wiki/entities/nvidia-sonic.md)、[paper-twist2](../../wiki/entities/paper-twist2.md)、[SceneBot](../../wiki/entities/paper-scenebot.md)、[CMP](../../wiki/entities/paper-cmp.md)、[BFM-Zero](../../wiki/entities/paper-bfm-zero.md)、[Humanoid-GPT](../../wiki/entities/paper-humanoid-gpt.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[运动跟踪选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)

## 当前提炼状态

- [x] arXiv HTML 方法 / Table 3 / 训练数据 / filter 摘录
- [x] 项目页开源核查（步骤 2.5）：Code coming soon
- [x] 升格 `wiki/entities/paper-gigabrain-wbc-0.5.md`
