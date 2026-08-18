# Seeker（arXiv:2608.13422）

> 来源归档（ingest）

- **标题：** Attention from Action, for Action: Emergent Visual Bottlenecks for Policy Learning
- **短名：** Seeker
- **类型：** paper / imitation-learning / visual-attention / ROI / diffusion-policy
- **arXiv：** <https://arxiv.org/abs/2608.13422>
- **PDF：** <https://arxiv.org/pdf/2608.13422>
- **HTML：** <https://arxiv.org/html/2608.13422>
- **代码：** <https://github.com/zheyu-zhuang/seeker> — [`sources/repos/seeker.md`](../repos/seeker.md)
- **作者：** Zheyu Zhuang、Ruiyu Wang、Florian T. Pokorny、Danica Kragic（KTH）；Nick Heppert、Abhinav Valada（弗莱堡大学）；Johannes Fabian Hahn（汉堡大学）
- **机构：** 瑞典皇家理工学院（KTH）；弗莱堡大学（University of Freiburg）；汉堡大学（Universität Hamburg）
- **版本：** arXiv:2608.13422（2026-08-14）；README bibtex 标 CoRL 2026
- **入库日期：** 2026-08-15
- **一句话说明：** 只用动作监督、不靠 gaze / 物体框 / VLM grounding，从冻结 DINOv3 patch 上学出随任务进度变化的 ROI；同一接口可做 RGB 裁剪、mask 引导增强与点云过滤。官方仓 **已开源、可运行**（MIT）。

## 摘要级要点

- **问题：** 视觉瓶颈能把「看哪」和「怎么动」拆开，但现成 ROI 要么靠外部空间标签（gaze、物体类、affordance），要么用夹爪/停顿关键帧把固定框钉在末端投影上。后者在连续运动、接触点偏离 TCP、关键帧噪声时会对不齐。
- **方法：** 任务 + 本体条件 query，在冻结 DINOv3 上做 **T 步门控多头 readout**；扩散动作头只作监督，训完丢掉。`top_p` 把注意力质量心变成框与粗 mask。
- **接口复用：** 第三人称 RGB 裁剪（框位置/尺度再 FiLM 回策略）；mask 引导 overlay；图像平面 ROI 过滤点云后再进 DP3。腕部相机不裁。
- **仿真（MimicGen，100 demo）：** 同栈最强输入级基线 RVT2-Crop 平均 **42.6%** → Seeker **62.6%**（相对 +52.8%）；接近特权 Oracle ROI **64.2%**，高于外部 SOTA RAVEN **52.1%**。
- **真机（xArm7，三任务 × 20 rollout）：** 域内平均 **76.7%** vs 最强基线 48.3%；光照/背景偏移平均 **60.0%** vs 20.0%；保留率 78.2%。
- **开源（截至 2026-08-15）：** `zheyu-zhuang/seeker` 默认分支 `open_source`；`seeker` CLI + MimicGen 权重；**已开源、可运行**。

## 核心摘录（面向 wiki 编译）

### 架构

1. **条件 query：** \(Q^{(0)}=\mathrm{FiLM}(Z,s)\)，\(s\) 用末端平移 + 夹爪；多本体要 embodiment id。任务嵌入来自 CLIP 短描述，只当标识。
2. **迭代门控 readout：** 每步 MHA 出 \(\{A_h,C_h\}\)，query 相关 \(\omega_h\) 融合后再 FiLM 更新 query。实验 **T=2**。
3. **粗到细 + trimming：** 先训 coarse；\(\tau\) epoch 后 fine 只看 coarse 框内 token；\(2\tau\) 后用 fine 框对 coarse 注意力做 KL trimming。推理只留 coarse。
4. **提取：** 丢掉得分最低的两头，`top_p=0.8` nucleus。Seeker 的 pooled context **不能**单独当策略（直接 rollout 接近 0）。

### 数字读法

| 设定 | Seeker | 对照 |
|------|--------|------|
| MimicGen 六任务均（100 demo） | **62.6%** | RVT2-Crop 42.6；Oracle 64.2；RAVEN 52.1 |
| 3-P Assembly / Threading 相对增益 | +119.9% / +83.6% | 接触/空间歧义任务最大 |
| DP3 + 手工工作区 + Seeker 过滤（200 demo，三任务） | PickPlace 26.0 / Stack-3 47.3 / Coffee 64.3 | 仅手工裁 1.3 / 23.3 / 22.0 |
| 真机 ID / OOD | **76.7 / 60.0** | RVT2-Crop 48.3 / 20.0 |
| 消融 | FiLM-only 39.4；低分辨率裁 61.2 | 收益主要来自显式裁剪，不是只告诉网络框在哪 |
| 监督目标 vs 扩散 ROI IoU | flow 0.67；IMLE 0.50；BC 0.30 | 生成式目标更稳 |

### 开源核查（步骤 2.5）

无独立 `*.github.io` 项目页；以论文 Code 链与用户给出的 GitHub 为准。仓库核查见 [`sources/repos/seeker.md`](../repos/seeker.md)：MIT、`seeker setup` / `train` / `rerender-dataset`、发布 `seeker.mimicgen.pth` → **已开源、可运行**。

## 对 wiki 的映射

- 升格 [Seeker 论文实体](../../wiki/entities/paper-seeker.md)
- 交叉：[Diffusion Policy](../../wiki/methods/diffusion-policy.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[MimicGen](../../wiki/entities/mimicgen.md)、[ActFovea](../../wiki/entities/paper-actfovea.md)、[接触–预测–适应 10 篇技术地图](../../wiki/overview/contact-predict-adapt-10-papers-technology-map.md)

## 当前提炼状态

- [x] 方法 + 仿真/真机数字 + 开源入口
- [x] wiki 实体、时序图与交叉引用
- [x] `sources/repos/`
