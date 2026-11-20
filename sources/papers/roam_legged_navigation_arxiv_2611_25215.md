# ROAM — Bridging vision-language reasoning and learned control for legged navigation in the wild（视觉语言推理与学习控制协同的野外腿式导航）

> 原始资料归档（paper），对应 [唯一论文节点](../../wiki/entities/paper-roam-open-world-legged-navigation.md)。

- **标题：** Bridging vision-language reasoning and learned control for legged navigation in the wild
- **作者：** Takahiro Miki, Robin Schmid, Manthan Patel, Turcan Tuna, Julia Richter, Julian Nubert, Joonho Lee, Matthias Müller, Marco Hutter
- **机构：** 苏黎世联邦理工（ETH Zurich；作者 1–7、9）；苹果公司（Apple；作者 8）
- **出版：** Science Robotics 11(120), eaee1549，2026-11-18，DOI [10.1126/scirobotics.aee1549](https://doi.org/10.1126/scirobotics.aee1549)
- **arXiv：** [2611.25215v1](https://arxiv.org/abs/2611.25215)（2026-11-18）；[HTML 正文及补充材料](https://arxiv.org/html/2611.25215v1)；[PDF](https://arxiv.org/pdf/2611.25215)
- **项目：** [roam-hiking.github.io](https://roam-hiking.github.io/)（[项目页归档](../sites/roam-hiking.md)）
- **代码：** [leggedrobotics/roam](https://github.com/leggedrobotics/roam)（[代码归档](../repos/leggedrobotics-roam.md)）；公开语义推理/分割/ROS 接口与高程图演示，非完整真机控制源码
- **数据：** [Zenodo 10.5281/zenodo.23492871](https://doi.org/10.5281/zenodo.23492871)（正文“Data, code, and materials availability”）
- **入库日期：** 2026-11-20
- **核对范围：** 对照 arXiv v1 HTML 论文引言（1）、结果（2.1–2.4，图 2–6、表 1）、讨论（3）、材料与方法（4.1–4.9）、末尾补充材料中记忆、实地介入、定位、VLM 对比/提示消融与楼梯高度实验及表 S2–S6；核实作者项目页与官方 GitHub README、`vlm_nav/example.py`、ROS 2 launch/config。Science.org 期刊正文访问受限，使用作者公开的 arXiv 版本核对。
- **通读：** 已完成

## 核心摘录（原文数字与边界）

1. **主问题（引言、图 1）：** 四足已有强运动能力，难在“选择哪里走”。VLM 不直接输出速度/坐标，而只输出环境类别及风险值；像素语义与几何规划交给下游，避免要求 VLM 精确空间推理。
2. **三张语义场（4.4）：** 对每个语义标签生成 `risk`（0 安全—1 危险）、`steppable`（0 不可落脚—1 可落脚）、`preferred_area`（0 非首选—1 首选）。SAN 开放词汇分割输出标签概率，经 `field_value = Σ(v_label × p_label)` 组合到像素值；“other” 标签有独立默认值。前 3 次 VLM 输出与 memory 文本参与平滑/上下文。
3. **异步频率（图 1、4.4）：** 原始实地部署 VLM 每 10 s 一次（0.1 Hz）、分割 6 Hz、GPU 高程图 20 Hz、导航策略 10 Hz、locomotion 50 Hz。另设 YOLO 标识牌检测：遇到牌子暂停并触发 VLM 关注。
4. **高程地图（4.4）：** LiDAR/立体深度 + 语义字段融合为 2.5D 高程地图，8×8 m、4 cm 分辨率。导航策略使用 elevation + risk + preferred_area，落足策略使用 elevation + steppable；语义信息在时间上融合而非每帧单独决策。
5. **Locomotion（4.5）：** RaiSim 仿真用 Perlin noise 生成可落脚/不可落脚区域，额外落错脚惩罚系数 α=0.1。privileged teacher → noisy student 蒸馏；对 steppable 输入不加大的噪声以免网络忽略语义。
6. **Local navigation（4.6）：** Isaac 风格仿真中生成可通行图、随机风险，RL 策略输出机身速度与转向。利用风险惩罚、preferred_area 对非首选区施加“虚拟增高”；8 m pure-pursuit lookahead。高层卫星地图人工选粗路点，OpenRouteService 生成 2–5 m 间隔路径，GNSS–LiDAR–IMU–运动学融合实现长距离对齐（4.7–4.8）。
7. **Zermatt（2.1）：** 10.1 km、下降 1220 m、最陡坡约 30%；时间自主率 96.7%，2 次因施工路段分割错误造成安全介入。
8. **Seealpsee（2.1、补充局限）：** 2.38 km 湖畔路径、20 m 高差；时间自主率 97.5%，3 次介入（湖边语义分割、滑坡边缘局部视野、无法穿越旋转门）。两次共 5 次主导根因：4 次 segmentation、1 次物理不可通过；不是“无人工介入”。
9. **受控测试（2.2–2.3）：** 1.2 km 场地对比 geometry-only 与 ViNT；施工标识牌触发绕行/回到道路。人体模型固定姿态 N=10：ROAM 10/10 到达且零碰撞；随机姿态仍 10/10 到达但 30% 轻微接触（补充 S4）。
10. **静态图像动作抽象（2.4、表 1）：** 27 张有效图片，ROAM + 图像空间 A* 的安全像素比例 0.991、首选区比例 0.868，27/27 到达；此处 A* 是为了隔离 VLM 空间表示而设置的离线代理，**不等同于实际运行的 RL 导航器**。共 30 张场景中的 3 张因 label pipeline 未输出而对所有方法排除。
11. **VLM/数据与边界（3、补充）：** 实地原始模型是 `gpt-4-vision-preview`，与仓库 README 当前默认的 Gemini 型号不是同一次实验设置；云端网络与服务依赖、少数实地样本、白天可见性、分割误判和厘米级空间推理差均为真实约束。补充楼梯案例中，15.5–16 cm 台阶会被一些模型在 15 cm 上限规则下误判可过。
12. **公开性（源码核对）：** [leggedrobotics/roam](https://github.com/leggedrobotics/roam) 覆盖前端及示例 `elevation_mapping_cupy` 对接；README 明示发布范围，没有完整 RL 策略训练/权重/ANYmal 全链路的公开复现实验指令。

## 对 wiki 的映射

- [paper-roam-open-world-legged-navigation](../../wiki/entities/paper-roam-open-world-legged-navigation.md)
- [视觉语言导航任务](../../wiki/tasks/vision-language-navigation.md)：与坐标/离散动作输出的不同抽象层对照
