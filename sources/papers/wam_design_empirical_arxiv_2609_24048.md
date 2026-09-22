# What Matters in Designing World Action Models: An Empirical Study

> 来源归档（ingest）

- **标题：** What Matters in Designing World Action Models: An Empirical Study
- **类型：** paper / world-action-model / empirical-study / controlled-ablation / manipulation
- **出处：** arXiv 预印本，2026-09（[2609.24048](https://arxiv.org/abs/2609.24048)）
- **论文链接：** <https://arxiv.org/abs/2609.24048>
- **PDF：** <https://arxiv.org/pdf/2609.24048>
- **作者：** Chao Tang *、Haoqing Wang *、Zilang Cen、Weishi Mi、Wei Xia、Fangcheng Liu、Anda Cheng、Yeqing Shen、Xiaohui Cui、Xiaoyuan Zhang、Yehui Tang、Tingguang Li（* 同等贡献）
- **机构：** 三星机器人体验（Samsung Robotics eXperience）；三星北京研发中心（Samsung R&D Institute China–Beijing）；北京大学（PKU）；武汉大学（WHU）；北京中关村学院（Zhongguancun Academy）
- **项目页：** **无**（截至 2026-09-22：arXiv / HTML 版均未列独立项目页或 Code 链接）
- **入库日期：** 2026-09-22
- **一句话说明：** 在 **固定骨干与训练管线** 下，对 WAM 的 **视频–动作因果结构（6 种）**、**潜空间世界表征（8 种 / 4 族）** 与 **世界–动作训练目标（4 种）** 做受控消融；RoboCasa-GR1（ID）、LIBERO / LIBERO-Plus（OOD）与 DROID 离线动作预测交叉验证。

## 开源状态（步骤 2.5，2026-09-22）

| 组件 | 状态 |
|------|------|
| 项目页 | **无** |
| GitHub / 权重 | **未见** 公开链接 |
| 实验框架引用 | 因果结构基于 [Fast-WAM（arXiv:2603.16666）](https://arxiv.org/abs/2603.16666) 族；潜空间与目标基于 [LDA-1B](../papers/sun_awesome_wm_2602_12215_lda-1b-scaling-latent-dynamics-action-mo.md) 族（论文正文引用） |

**结论：确认未开源** — 受控研究论文，截至入库日无复现入口。

## 核心摘录（面向 wiki 编译）

### 1) 视频–动作因果结构（6 种，Fast-WAM 框架内对照）

- **Disentangled/Unconditional、Video-to-Action、Action-to-Video、Bidirectional、Joint、Causally Interleaved**
- **Takeaway 1.1：** 生成未来主要通过 **时间组织** 影响动作；**内容 corruption** 对动作预测与成功率影响极小，**时间 reversal** 在 OOD 上降幅可达 **24–32 pp**。
- **Takeaway 1.2：** **Causal video generation** 比严格的 video–action token 时序因果更关键；Causally Interleaved 在 LIBERO-Plus **77.33%**；Video-Causal Global 变体 **79.84%**。
- **结构对比：** Joint vs Uncond、Bidirectional vs Action-to-Video 在 LIBERO-Plus 上 route-enabled 变体分别 **+16.14% / +14.93%**，但在 ID RoboCasa-GR1 上反而 **−2.33% / −4.00%**。

**对 wiki 的映射：** [`wiki/entities/paper-wam-design-empirical-study.md`](../../wiki/entities/paper-wam-design-empirical-study.md)

### 2) 潜空间世界表征（8 种 / 4 族，LDA-1B 框架内对照）

- **Semantic：** DINOv3、Qwen3-VL、SAM3
- **Geometric：** Depth Anything 3、VGGT-Ω
- **Reconstructive：** Image-VAE、Video-VAE
- **Predictive：** V-JEPA 2.1
- **Inter-frame vs framewise：** DA3 / VGGT-Ω / V-JEPA / Qwen3-VL(video) / Video-VAE 为 **跨帧 inter-frame**；DINOv3 / SAM3 / Image-VAE 为 **逐帧 framewise**。
- **Takeaway 2.1：** inter-frame 在 **ID RoboCasa-GR1** 更优；**OOD LIBERO-Plus** 排序反转，framewise 在 sensor noise / camera viewpoint 等扰动上 margin 最大。
- **Takeaway 2.2：** 线性 probe 显示 ID 上 inter-frame 动作更可解码；OOD 上 framewise 领先；扰动历史/当前帧配对时 inter-frame 退化更剧烈。

**对 wiki 的映射：** 同上实体页 + 交叉 [`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)

### 3) 世界–动作建模目标（4 种，LDA-1B 框架内对照）

- **BC** \(p(a_{t+1:t+k}\mid o_t,\ell)\)、**IDM**、**FDM**、**VG**（video generation）
- **Takeaway 3.1：** **ID** 上 BC-only 最高；辅助目标普遍 **分流容量**；**OOD** 上 **BC+VG** 从 **77.96%→81.22%**（camera viewpoint **+13.32%**）；FDM 边际；IDM 略降。
- **Takeaway 3.2：**  naive joint 多目标 **−4.04%** vs BC+VG；**分阶段**（前 80% BC+VG，后 20% 引入 dynamics）LIBERO-Plus **83.15%** 最高；同策略 **不改善 ID**。

**对 wiki 的映射：** 同上实体页

### 4) DROID 真机数据离线验证（200K steps，matched settings）

| 轴 | 对照 | 读法 |
|----|------|------|
| 因果 | Uncond vs Causally Interleaved | 后者 MSE/L1 更低、Accuracy@0.1/0.5 更高 |
| 表征 | DA3 vs DINOv3 | framewise DINOv3 四项指标均优 |
| 目标 | BC-only vs BC+VG | BC+VG 全面优于 BC-only |

**对 wiki 的映射：** 同上实体页「DROID 验证」节

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-wam-design-empirical-study.md`](../../wiki/entities/paper-wam-design-empirical-study.md)**
- 概念交叉：**[`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)**
- 方法/任务：**[`wiki/methods/vla.md`](../../wiki/methods/vla.md)**、**[`wiki/tasks/manipulation.md`](../../wiki/tasks/manipulation.md)**
- 相邻 WAM 实例：**[`wiki/entities/paper-glancewam.md`](../../wiki/entities/paper-glancewam.md)**（Fast-WAM 系部署）、**[`wiki/entities/paper-effvla.md`](../../wiki/entities/paper-effvla.md)**（VLA head 受控研究对照）
