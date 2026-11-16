# RoboEdit（arXiv:2608.18948）

> 来源归档（ingest）

- **标题：** RoboEdit: Turning Human Manipulation Videos into Scalable Robot Experience
- **类型：** paper / robot-data / human-video / video-editing / cross-embodiment / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.18948>
- **PDF：** <https://arxiv.org/pdf/2608.18948>
- **HTML：** <https://arxiv.org/html/2608.18948>
- **项目页：** <https://roboedit.github.io/>（见于 arXiv v4 摘要页元数据；v4 PDF 正文未含 GitHub/HF 仓库 URL）
- **机构：** ¹ 加州大学洛杉矶分校（UCLA）；² 麻省理工学院（MIT）；³ 犹他大学（University of Utah）（通讯 Demetri Terzopoulos、Ying Jiang、Chenfanfu Jiang，均属 UCLA）
- **作者：** Yaowei Guo¹、Zeng Tao¹、Yuxin Jiang¹、Yunuo Chen¹、Zhiyang Dou²、Yuxiang Ma²、Yin Yang³、Demetri Terzopoulos¹†、Ying Jiang¹†、Chenfanfu Jiang¹†
- **发表 / 上传：** 2026-08-19（arXiv v1）；2026-10-17（arXiv v4）
- **训练栈：** Nova / NovaEdit（`arXiv:2603.02802`）思路扩展，初始化自 `Wan2.1-Fun-1.3B-InP` + `Wan2.1-T2V-1.3B`（文本编码器与 VAE）；flow matching；LoRA（2.5M）+ 末 10 层残差瓶颈适配器（7.9M）；`Qwen-Image-Edit` 生成推理关键帧；ResNet-34 FPN + SQPnP + 81 帧时序 Transformer 3D 解码器
- **入库日期：** 2026-08-21（2026-11-14 核对 v4 全文与附录）
- **核对范围：** 2026-11-14 通读 arXiv:2608.18948v4 全文（摘要、§1–§5、Tables 1–4、Figs 1–8、Refs [1]–[48] 及全部附录 A–H 含 Tables S1–S3、Figs S1–S6），并核对 `2608.18948` arXiv 摘要页元数据（确认 `https://roboedit.github.io/`）；项目页未做离线抓取，亦无公开代码仓。
- **通读：** 已完成

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.18948v4](https://arxiv.org/abs/2608.18948) | 正文 5 节 + 附录 A–H（22 页） |
| 项目页 | <https://roboedit.github.io/> | arXiv v4 abstract 元数据所列项目主页 |
| 数据集 | RoboEdit-14M | 174,547 clips / 14,138,307 frames（30 FPS 下 130.91 h）/ 7 embodiments（Table S1–S2） |
| 源数据 | DexYCB、HOT3D、H2O、GigaHands、TACO | RoboEdit-ADC 输入（24,197 源 clip）+ 29,088 RoboEngine 合成对 |
| 基线 | VACE、UniVideo、VINO、Kiwi-Edit、OmniWeaving、EditCtrl、AnyV2V、ReCo | 300-case benchmark（Appendix H） |

## 开源状态（步骤 2.5，2026-11-14）

- **代码与数据集未在文中公开：** arXiv v4 摘要页补充了项目主页 <https://roboedit.github.io/>，但 `2608.18948v4` 论文全文、图表与附录均 **未提供** GitHub 仓库、Hugging Face 数据集/权重下载链接。
- **处理：** wiki 标「仅见项目主页、代码与 RoboEdit-14M 暂未公开」；`## 源码运行时序图` 写 **不适用**，待官方在主页/GitHub 发布源码与数据集后再补 `sources/repos/`。

## 摘要级要点

- **问题：** 机器人 hand-object 视频采集成本高且具身绑定；人类操作视频丰富但 morphology/kinematics 不匹配，难直接作 robot 训练数据。
- **RoboEdit 三件套：**
  - **RoboEdit-ADC** — Appendix A 自动化 paired 数据生成管线：RGB → 3D HOI 重建 \(\mathcal{Z}_{1:T}=(H,O,M_o,m,C)\) → depth regularization + physics-refined retargeting \(q^e_{1:T}\) → human/object inpaint + render/composite → paired \((v^h, v^{r,e}, q^e)\)。
  - **RoboEdit-Trans** — 基于 `Wan2.1-Fun-1.3B-InP` 与 `Wan2.1-T2V-1.3B` 初始化（283.4M，10K steps），结合 LoRA（2.5M）与后 10 blocks residual bottleneck adapter（7.9M）实现跨具身视频编辑；配套 **3D Robot-State Decoder**（ResNet-34 FPN 2D heatmap anchors + SQPnP wrist + 81-frame temporal Transformer refine → FK trajectory）恢复逐帧状态。
  - **RoboEdit-14M** — Table S1–S2 统计共 `174,547` clips / `14,138,307` frames / `130.91 h`（30 FPS）；覆盖 7 embodiments（Inspire、XHand、Ability、SCHUNK SVH、Allegro、Unitree Dex3、Franka Panda gripper）。
- **编辑与推理结果（Table 2 / Table 4 / Table S3）：**
  - Table 2（300-case benchmark）：`Ours M` 达到 SSIM `0.9282`、LPIPS `0.0470`、Edit LPIPS `0.0171`、BG SSIM `0.9188`、VBench AQ `0.4832` / DD `0.6300` / MS `0.9956`、OpenVE `3.2511`（注：`VINO S` 在 OpenVE `3.2711` 与 MS `0.9967` 上略高，并非全指标第一）。
  - Table 4（3D Decoder，`Full`）：Wrist `42.03 mm`、Rot `23.28°`、Fingertip `43.26 mm`、Joint `0.2260`。
  - Table S3（H100 NVL 单样本推理耗时）：ADC 管线 `1199.94 s`，而编辑推理仅 Qwen `267.71 s` + Trans `231.28 s` = `498.99 s`。
- **下游控制（§4.5 & Appendix G）：** Genesis 512 并行环境下用 residual PPO 跟踪 decoded trajectory，Panda 达 `71%`、XHand 达 `62%` 仿真成功率；Franka Panda 真机 YCB 4 任务每项 10 次试验均达 `>=50%` 成功率（Fig. 8）。

## 核心摘录（面向 wiki 编译）

### 1) RoboEdit-ADC 管线（Appendix A & Fig. 7）

重建 \(\mathcal{Z}_{1:T}=(H,O,M_o,m,C)\) → retarget \(q^e_{1:T}\) → inpaint human/object → render robot under \(C_{1:T}\) → paired \((v^h, v^{r,e}, q^e)\)。

Depth regularization 对齐 HaMeR 尺度；physics refinement 减少 floating / penetration（Fig. 7 展示跨具身 retargeting 效果）。

### 2) RoboEdit-Trans 与 3D Decoder

- Backbone：`Wan2.1-Fun-1.3B-InP` + `Wan2.1-T2V-1.3B` 初始化（283.4M，10K steps）；81-frame clips；sparse keyframes \(\{0,10,\ldots,80\}\)。
- Cross-embodiment：LoRA（2.5M）+ 后 10 blocks residual bottleneck adapter（7.9M）。
- 3D Decoder：ResNet-34 FPN 2D heatmap anchors + SQPnP wrist + 81-frame temporal Transformer refine → FK trajectory。
- 对比锚点：`Nova` 引用为 Pan et al. 2026b（`arXiv:2603.02802`，非 `2506.07120`）。

### 3) 消融（Table 3）

LoRA + Adapter 联合最优；单独 residual adapter 增益大于单独 LoRA。

### 4) 与 Ego2Robot 等对照

输出是 **full robot interaction video + dense 3D states**，而非仅 EEF 轨迹或中间 representation。

## 对 wiki 的映射

- 沉淀实体页：[RoboEdit](../../wiki/entities/paper-roboedit.md)
- 交叉补强：[Ego2Robot](../../wiki/entities/paper-ego2robot.md)、[motion retargeting](../../wiki/concepts/motion-retargeting.md)、[manipulation 任务](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] `arXiv:2608.18948v4`（2026-10-17）正文与附录全篇核读（Table 2/3/4/S1–S3、Fig. 7–8、Appendix A/G）
- [x] 开源与勘误核查：v4 `.abs.html` 增补 `https://roboedit.github.io/`，论文正文及附录未提供代码/数据集仓库 URL；修正此前误标 UPenn（实为 UCLA / MIT / University of Utah）与 `Nova` 错误 arXiv 编号（实为 `2603.02802`）
- [x] 升格 `wiki/entities/paper-roboedit.md`
