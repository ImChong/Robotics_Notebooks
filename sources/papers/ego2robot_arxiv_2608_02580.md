# Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data（arXiv:2608.02580）

> 来源归档（ingest）

- **标题：** Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data
- **缩写 / 框架：** **Ego2Robot**
- **类型：** paper / robot-data / egocentric / vla / synthesis
- **arXiv：** <https://arxiv.org/abs/2608.02580>
- **项目页：** <https://www-ye.github.io/ego2robot_blog/>（归档见 [`sources/sites/ego2robot-blog.md`](../sites/ego2robot-blog.md)）
- **作者：** Ye Wang、Pei Lin、Xiong-Hui Chen、Haoqi Yuan 等（人大 AIM3 / 阿里 Qwen / 上科大 / BIGAI 等）
- **机构：** 中国人民大学（RUC）；阿里巴巴通义千问（Alibaba Qwen）；上海科技大学（ShanghaiTech）；北京通用人工智能研究院（BIGAI）；北京航空航天大学（BUAA）
- **入库日期：** 2026-08-15
- **一句话说明：** 把第一人称人类操作视频经动作重定向、机械臂视觉合成与三级质检，做成 **15 种形态、18,561 小时** 机器人训练数据；与真机数据联合预训练提升 VLA 的解耦 OOD，并在 ARX ACone 真机验证。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-15）：** [www-ye.github.io/ego2robot_blog](https://www-ye.github.io/ego2robot_blog/) 有管线、数据配比、RoboTwin 解耦评测与真机表；配套仓 [www-Ye/ego2robot_blog](https://github.com/www-Ye/ego2robot_blog) **仅静态站**（`index.html` + 图），**无训练脚本、无数据下载**。
- **结论：** **项目页已发布；合成管线与 18,561 h 数据未开源。** 源码运行时序图标 **不适用**。文内称该数据进入 Qwen-RobotManip 预训练配方（arXiv:2606.17846）。

## 摘录 1：管线

| 阶段 | 要点 |
|------|------|
| **输入** | Path A：已标注手姿态；Path B：WiLoR 逐帧 + DynHaMR 时序；长视频用 Qwen3.5 切子任务 |
| **动作对齐** | 21 关键点 → TCP / 开合 / 抓取坐标系；Savitzky–Golay + SLERP；按源降采样对齐机器人速度 |
| **视觉对齐** | SAM 3 臂分割 → ProPainter 去手 → 基座位姿网格搜索 + MuJoCo IK → 深度合成 |
| **质检** | L1 管线内（IK/碰撞/离群）；L2 统计；L3 VLM 语义一致性 |
| **动作表示** | 相机系相对 EEF，避免未知外参下的世界系不兼容 |

源：ANT 7h + EgoDex 732h + ViTRA 249h + EgoVerse 954h ≈ 1,940 h → ×15 形态 → **18,561 h**；共训真机约 6,565 h（DROID / AgibotWorld / InternData）。

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
