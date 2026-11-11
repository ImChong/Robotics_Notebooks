# Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data（arXiv:2608.02580）

> 来源归档（ingest）

- **标题：** Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data
- **缩写 / 框架：** **Ego2Robot**
- **类型：** paper / robot-data / egocentric / vla / synthesis
- **arXiv：** <https://arxiv.org/abs/2608.02580v1>（v1，2026-08-03；[完整 HTML](https://arxiv.org/html/2608.02580v1)）
- **项目页：** <https://www-ye.github.io/ego2robot_blog/>（归档见 [`sources/sites/ego2robot-blog.md`](../sites/ego2robot-blog.md)）
- **作者：** Ye Wang、Pei Lin、Xiong-Hui Chen、Haoqi Yuan 等（人大 AIM3 / 阿里 Qwen / 上科大 / BIGAI 等）
- **机构：** 中国人民大学（RUC）；阿里巴巴通义千问（Alibaba Qwen）；上海科技大学（ShanghaiTech）；北京通用人工智能研究院（BIGAI）；北京航空航天大学（BUAA）
- **入库日期：** 2026-08-15
- **重新核对日期：** 2026-11-12
- **代码一手来源：** [www-Ye/ego2robot_blog](../repos/ego2robot-ye-code.md)
- **一句话说明：** 把第一人称人类操作视频经动作重定向、机械臂视觉合成与三级质检，做成 **15 种形态、18,561 小时** 机器人训练数据；与真机数据联合预训练提升 VLA 的解耦 OOD，并在 ARX ACone 真机验证。

## 原文阅读记录（2026-11-12）

- **核对范围：** 通读 arXiv HTML v1 正文 §1–7，核对 Fig. 1–5 的流程和图注、表 1–2 的主实验/扰动指标；通读附录 A.1–A.7 手部重建、动作与视觉对齐、L1–L3、15 种形态；附录 B.1–B.4 扰动协议与任务、附录 C.1–C.3 相机系相对 EEF 与 flow matching、附录 D.1–D.4 训练配置与数据/消融、附录 E.1–E.4 ACone 真实实验及分步评分、附录 F.1–F.2 π₀.₅ 对照与 50 项逐任务结果。参考文献核对题目及主要来源引用关系，未逐项验证每篇外部论文；附录 F 的逐任务表核对汇总口径，未独立重算全部实验。
- **通读：** 已完成
- **独立复现：** 尚未运行模型或 GPU 管线，也未下载权重和数据；此处仅确认论文描述与公开代码执行入口，论文数值是作者报告，不是本次复测。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-15）：** [www-ye.github.io/ego2robot_blog](https://www-ye.github.io/ego2robot_blog/) 有管线、数据配比、RoboTwin 解耦评测与真机表；配套仓 [www-Ye/ego2robot_blog](https://github.com/www-Ye/ego2robot_blog) **仅静态站**（`index.html` + 图），**无训练脚本、无数据下载**。
- **更新（2026-11-12）：** [官方原仓](https://github.com/www-Ye/ego2robot_blog) 现在含 `ego2robot/`、`scripts/run_pipeline.py`、`configs/` 与 `evaluation/robotwin/`，公开合成管线与评测框架；原「只有静态站/未开源」为**历史状态，已失效**。模型权重、原始数据/镜像、专有模型及环境仍需按 README 单独准备，不声称 18,561 h 全量成品数据可公开下载。[源码静态核对归档](../repos/ego2robot-ye-code.md) 提供入口时序图依据。文内称该数据进入 Qwen-RobotManip 预训练配方（arXiv:2606.17846）。

## 摘录 1：管线

| 阶段 | 要点 |
|------|------|
| **输入** | Path A：已标注手姿态；Path B：WiLoR 逐帧 + DynHaMR 时序；长视频用 Qwen3.5 切子任务 |
| **动作对齐** | 21 关键点 → TCP / 开合 / 抓取坐标系；Savitzky–Golay + SLERP；按源降采样对齐机器人速度 |
| **视觉对齐** | SAM 3 臂分割 → ProPainter 去手 → 基座位姿网格搜索 + MuJoCo IK → 深度合成 |
| **质检** | L1 管线内（IK/碰撞/离群）；L2 统计；L3 VLM 语义一致性 |
| **动作表示** | 相机系相对 EEF，避免未知外参下的世界系不兼容 |

源：ANT 7h + EgoDex 732h + ViTRA 249h + EgoVerse 954h ≈ 1,940 h → ×15 形态 → **18,561 h**；共训机器人数据约 6,565 h（DROID / AgibotWorld / InternData；其中 InternData 为仿真数据，并非全部真实机器人）。

## 摘录 2：评测协议

扩展 RoboTwin 2.0 为四轴独立扰动：视觉外观、场景布局、本体形态、任务语义（未见物体 + 505 条改写指令），外加更高机位的 EBench。策略：Qwen3.5-4B + DiT 动作头，32 步相机系相对 EEF chunk。

## 摘录 3：数字（项目页表）

| 预训练 | Clean | Rand | Visual | Scene | Embody | Task | EBench |
|--------|------:|-----:|-------:|------:|-------:|-----:|-------:|
| Robot-only | 62.2 | 50.9 | 61.4 | 52.9 | 23.8 | 46.2 | 39.6 |
| Ego2R+Robot 1:1 | **68.1** | **53.5** | **67.3** | **56.9** | 27.2 | **54.1** | 49.8 |
| Ego2R+Robot 3:1 | 64.1 | 49.2 | 62.7 | 54.3 | **28.2** | 51.6 | **51.7** |

- 未见物体 29.3→40.0（3:1）；语言改写 63.1→68.5（1:1）；UR5 20.2→31.4（3:1）；Franka 仍 <7%。
- 仅 ego 预训练：生视频 28.1 → 单形态管线 31.7 → 15 形态 33.5 → 再加生视频 37.3。
- 真机 ARX ACone、每任务 20 条遥操作：Mix + Ego2R Play（现场 ego-play 约 7 min）五任务全最高。

**对 wiki 的映射：** [`wiki/entities/paper-ego2robot.md`](../../wiki/entities/paper-ego2robot.md)；交叉 [VLA](../../wiki/methods/vla.md)、[EgoScale](../../wiki/methods/egoscale.md)、[WiLoR](../../wiki/methods/wilor.md)、[RoboTwin](../../wiki/entities/robotwin.md)、[EgoVerse](../../wiki/entities/paper-egoverse.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（仅项目页）

## 补充：原文附录的工程与评测边界（2026-11-12）

- **动作映射（正文式 1–3，附录 A.3）：** 虚拟指尖为食指尖 ×0.7 + 中指尖 ×0.3；TCP 为拇指尖与虚拟指尖的中点，宽度为两者距离。位置/宽度用 Savitzky–Golay，方向用 Gaussian-weighted SLERP；退化抓取方向沿用最近有效解。
- **视觉与筛选（式 4–5，附录 A.4–A.5）：** MuJoCo IK 找可达 base placement；候选评分含可达率与 normalized reach 65% 附近的偏好。SAM 3 + ProPainter 去手，机器人合成时按深度处理夹爪遮挡。L1 追踪误差阈值 0.05 m、碰撞等；L2 剔除极端值/突变及 invalid 比例超过 60% 的 episode；L3 让 VLM 判断视频–动作语义一致性。
- **网络/动作头（附录 C）：** Qwen3.5-4B 视觉语言编码 + DiT 动作头，动作是 7 维相对 EEF（位置、旋转向量、夹爪），chunk 32；训练 8 个 flow/diffusion 步，推理 4 Euler 步（主文「8 diffusion steps」为训练配置，不能误说推理需要 8 步）。
- **公平比较（附录 D）：** 8×A100、200K 预训练 step、GPU batch 12、共约 19.2M frames；数据比 1:3/3:1/1:1 为采样权重而非固定每个原始 dataset 大小。RoboTwin Clean 50 任务×50 demo 微调，50K steps；EBench 7 任务×400 demo 独立微调。
- **真机试验（附录 E）：** ARX ACone 双臂每臂 6 DOF、15 Hz 控制；5 任务×每任务 20 条遥操作，约 35 min ego-play 变为 675 段合成 episode，三设置分别 Robot-only、Mix、Mix+Ego2R Play。评分为**任务子步骤的加权部分分**（每任务满分 100、每设置 20 次 trial），不能将图中的“得分”直接称完整任务成功率。
- **局限（§6）：** 平行夹爪忽略精细手指关节；遮挡/复杂光照下 inpaint/depth 会产生伪影；评估范围以 RoboTwin 2.0 和所选真机任务为主。

**原文链接：** [arXiv HTML v1](https://arxiv.org/html/2608.02580v1)；**可运行源码：** [官方仓](https://github.com/www-Ye/ego2robot_blog)；[代码阅读归档](../repos/ego2robot-ye-code.md)。
