# DreamWAM（arXiv:2608.04996）

> 来源归档（ingest）

- **标题：** DreamWAM: Beyond RGB Future Prediction for World Action Models
- **类型：** paper / world-action-models / joint-wam / manipulation / libero
- **arXiv abs：** <https://arxiv.org/abs/2608.04996>
- **PDF：** <https://arxiv.org/pdf/2608.04996>
- **HTML：** <https://arxiv.org/html/2608.04996>
- **项目页：** <https://hustvl.github.io/DreamWAM/> — 归档见 [`sources/sites/hustvl-dreamwam-github-io.md`](../sites/hustvl-dreamwam-github-io.md)
- **代码：** <https://github.com/hustvl/DreamWAM> — 归档见 [`sources/repos/dreamwam.md`](../repos/dreamwam.md)
- **权重：** <https://huggingface.co/hustvl/DreamWAM>（MIT；`dreamwam_joint.pt` / `dreamwam_uncond.pt`）
- **机构：** 华中科技大学（HUST / hustvl）；地瓜机器人（D-Robotics）；武汉大学（WHU）；地平线（Horizon Robotics）
- **作者：** Shanglin Yuan\*、Weiheng Zhao\*、Xin Shi、Haoyi Jiang、Xianda Guo、Liu Liu、Wenyu Liu、Wei Sui†、Xinggang Wang‡（\* equal；† Project Lead；‡ Corresponding）
- **发表 / 上传：** 2026-08（arXiv:2608.04996）
- **入库日期：** 2026-08-07
- **一句话说明：** Joint WAM 把未来预测从「仅 RGB」扩成 appearance / motion / geometry / semantics 结构化世界建模；训练用 RAFT flow、DA3 depth、DINOv2 门控残差监督，推理仍 RGB-only；相对 Fast-WAM-Joint 在 LIBERO-Plus 与真机扰动上显著更稳。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [hustvl.github.io/DreamWAM](https://hustvl.github.io/DreamWAM/) | 方法图、LIBERO / 真机表与 rollout |
| 代码 | [hustvl/DreamWAM](https://github.com/hustvl/DreamWAM) | 训练 / 评测 / 预处理；基于 FastWAM |
| 权重 | [hustvl/DreamWAM](https://huggingface.co/hustvl/DreamWAM) | joint / uncond checkpoint |
| 基线仓 | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | VideoDiT–ActionDiT 耦合配方源头 |
| 数据 | [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) | 四套件 LeRobot v2.1 LIBERO |

## 开源状态（步骤 2.5，截至 2026-08-07）

- **已开源：** 项目页列出 Paper / Code / Models；GitHub 含 `scripts/train.py`、`eval_libero.py`、`eval_libero_plus.py`、`precompute_cache.py`；HF 权重可下。
- **许可证：** HF 卡标明 **MIT**；GitHub API 未返回 SPDX（截至核查日）。
- **处理：** wiki 写「已开源」并补 `## 源码运行时序图`；互链 `sources/repos/` 与 `sources/sites/`。

## 摘要级要点

- **问题：** 多数 WAM 在 RGB 空间预测未来，任务相关状态转移与纹理 / 光照 / 背景 / 视角 nuisance 缠在一起。
- **主张：** WAM 应显式预测 **action-relevant future state**，用 appearance + motion + geometry + semantics 互补视图，而不是只追像素还原。
- **方法：** VideoDiT 对 RGB 与 optical-flow latent **联合去噪**；DA3 depth 与 DINOv2 经 **轻量 gated residual** 注入选定 block；ActionDiT 与 VideoDiT **shared attention**；推理关闭 beyond-RGB 分支。
- **结果要点（项目页 / 摘要）：**
  - LIBERO joint video-action：Fast-WAM-Joint **98.00% → DreamWAM 98.90%**
  - LIBERO-Plus（未见扰动、无 Plus 训练）：**69.16% → 75.47%**
  - 真机视觉扰动平均：**55.6% → 74.40%**；标准任务 **90.8% → 96.7%**
- **局限：** 依赖离线预计算 RAFT / DA3 / DINO 目标与 Wan2.2 VAE 管线；评测主轴为桌面操纵 LIBERO 族 + 双臂真机，非全身 loco-manip。

## 核心摘录（面向 wiki 编译）

### 1) 结构化未来四视图

| 视图 | 监督源 | 角色 |
|------|--------|------|
| Appearance | RGB / Wan VAE latent | 场景演化外观 |
| Motion | RAFT → 伪彩 flow → 同 VAE | 时序变化 |
| Geometry | DA3 depth → PCA-8 | 空间结构 |
| Semantics | DINOv2 ViT-B/14 → PCA-8 | 物体级语义 |

### 2) 训练 vs 部署

- **训练：** RGB+Flow 联合去噪 + Depth/DINO 门控残差；动作支路吃共享注意力。
- **部署：** **仅 RGB 观测 → 动作**（beyond-RGB 分支全关）。

### 3) 复现入口（README）

```bash
accelerate launch --num_processes 8 scripts/train.py --config configs/dreamwam_joint.yaml
python scripts/eval_libero.py --config configs/dreamwam_joint.yaml --suite libero_spatial
python scripts/eval_libero_plus.py --config configs/dreamwam_joint.yaml
```

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-dreamwam.md](../../wiki/entities/paper-dreamwam.md)
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[DynaWM](../../wiki/entities/paper-dynawm-vla-online-correction.md)、[DreamSteer](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md)、[动作后果技术地图](../../wiki/overview/robot-world-models-action-consequence-technology-map.md)

## 全文通读复核（2026-11-13）

- **核对范围：** arXiv:2608.04996v1（2026-08-05）正文全部章节、附录 A–B（目标构造、训练/推理及真机协议），图表文字层；参考文献浏览。另读官方项目页全文、根 README 与下述固定版本源码；未运行训练或真机实验。
- **通读：** 已完成
- Flow 是 RAFT 光流伪彩视频再经同一 Wan VAE；不是直接输入度量三维运动。DA3 log-depth 与 DINO patch 特征分别用仅训练集拟合的 PCA-8；九帧压到首帧、中四帧平均、末四帧平均。
- 论文部署叙述为关闭几何/语义残差及教师、flow 通道置零；RGB-only 指传感输入，仍有本体状态和语言，不意味着 Joint 分支不生成未来。当前代码保留 residual injection，与论文叙述有差异，见仓库复核。
- Joint 对照 LIBERO 98.00→98.90、Plus 69.16→75.47（后者七扰动维度不加权平均），两 seed；Goal 98.8→98.6，不能称所有子任务均提升。Uncond 是另一设置，不能混用两者基线。
- 真机为双臂 AgileX PiPER；四标准任务各30次，三扰动仅 Strawberry Selection 各30次，共每模型210次。90.8→96.7与55.6→74.4的单位是成功率百分比，增益为5.9/18.8个百分点；三扰动不是三种新技能。
- 路由消融直接去噪全部视图97.85略低RGB98.0，混合D/R/R98.9；目标种类更多不自动更好。未验证接触力、度量3D或全身运动的物理一致性。
