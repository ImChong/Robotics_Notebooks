# RoboEdit（arXiv:2608.18948）

> 来源归档（ingest）

- **标题：** RoboEdit: Turning Human Manipulation Videos into Scalable Robot Experience
- **类型：** paper / robot-data / human-video / video-editing / cross-embodiment / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.18948>
- **PDF：** <https://arxiv.org/pdf/2608.18948>
- **HTML：** <https://arxiv.org/html/2608.18948v1>
- **机构：** 加州大学洛杉矶分校（UCLA）；宾夕法尼亚大学（UPenn，Demetri Terzopoulos）等（通讯 Ying Jiang、Chenfanfu Jiang）
- **作者：** Yaowei Guo、Zeng Tao、Yuxin Jiang、Yunuo Chen、Zhiyang Dou、Yuxiang Ma、Yin Yang、Demetri Terzopoulos、Ying Jiang、Chenfanfu Jiang
- **发表 / 上传：** 2026-08-19（arXiv v1）
- **训练栈：** NovaEdit / Wan2.1-VACE-1.3B backbone；flow matching；LoRA + residual adapter；Qwen-Image-Edit keyframe
- **入库日期：** 2026-08-21

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| arXiv | [2608.18948](https://arxiv.org/abs/2608.18948) | 论文与附录 |
| 数据集 | RoboEdit-14M | 174K pairs / 14.1M frames / 7 embodiments |
| 源数据 | DexYCB、HOT3D、H2O、GigaHands、TACO | RoboEdit-ADC 输入 |
| 基线 | VACE、UniVideo、VINO、Kiwi-Edit 等 | 300-case benchmark |

## 开源状态（步骤 2.5，2026-08-21）

- **确认未开源：** arXiv abs / HTML **无** 项目页、GitHub 或 Hugging Face URL；论文未给出 code/data 发布链接。
- **处理：** wiki 标未开源；RoboEdit-14M 待官方发布再补 `sources/repos/`。

## 摘要级要点

- **问题：** 机器人 hand-object 视频采集贵且具身绑定；人类操作视频丰富但 morphology/kinematics 不匹配，难直接当 robot 训练数据。
- **RoboEdit 三件套：**
  - **RoboEdit-Trans** — 在保留场景/相机/物体动力学下，把 human video **编辑** 为目标 robot video；LoRA + residual adapter 跨具身；**3D Robot-State Decoder** 恢复 per-frame hand states。
  - **RoboEdit-ADC** — RGB → 3D HOI 重建 → depth/physics-refined retarget → inpaint + composite；自动构造 paired supervision。
  - **RoboEdit-14M** — 174,547 aligned pairs（14.1M frames）；7 embodiments（Inspire、XHand、Ability、SCHUNK SVH、Allegro、Unitree Dex3、Franka Panda gripper）。
- **编辑结果（300-case benchmark）：** SSIM **0.9282**、Edit LPIPS **0.0171**、OpenVE **3.2511**（SOTA vs 8 baselines）。
- **下游控制：** Genesis 中 residual PPO 跟踪 decoded trajectory — Panda **71%**、XHand **62%** sim success；Franka Panda 真机 YCB 四任务。

## 核心摘录（面向 wiki 编译）

### 1) RoboEdit-ADC 管线

重建 \(\mathcal{Z}_{1:T}=(H,O,M_o,m,C)\) → retarget \(q^e_{1:T}\) → inpaint human/object → render robot under \(C_{1:T}\) → paired \((v^h, v^{r,e}, q^e)\)。

Depth regularization：对齐 HaMeR 尺度；physics refinement：减 floating / penetration。

### 2) RoboEdit-Trans

- Backbone：Wan2.1-VACE-1.3B；81-frame clips；sparse keyframes \(\{0,10,\ldots,80\}\)。
- Cross-embodiment：LoRA on spatiotemporal + residual bottleneck adapter on hand geometry。
- 3D Decoder：2D heatmap anchors + PnP wrist + temporal Transformer refine → FK trajectory。

### 3) 消融（Table 3）

LoRA + Adapter 联合最优；单独 residual adapter 增益大于单独 LoRA。

### 4) 与 Ego2Robot 等对照

输出是 **full robot interaction video + dense 3D states**，而非仅 EEF 轨迹或中间 representation。

## 对 wiki 的映射

- 沉淀实体页：[RoboEdit](../../wiki/entities/paper-roboedit.md)
- 交叉补强：[Ego2Robot](../../wiki/entities/paper-ego2robot.md)、[motion retargeting](../../wiki/concepts/motion-retargeting.md)、[manipulation 任务](../../wiki/tasks/manipulation.md)、[Sim2Real](../../wiki/concepts/sim2real.md)

## 当前提炼状态

- [x] arXiv HTML 方法 / Table 2–3 / RoboEdit-14M / 真机部署摘录
- [x] 开源核查：无项目页与代码 URL
- [x] 升格 `wiki/entities/paper-roboedit.md`
