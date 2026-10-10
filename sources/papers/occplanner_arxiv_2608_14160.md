# OccPlanner: Goal-Aware Occupancy-Conditioned Diffusion Planner for PixelGoal Navigation（论文来源）

> 来源归档（ingest）

- **类型：** arXiv 技术报告（cs.RO）
- **作者：** Binling Huang（第一作者）、Nianjin Ye、Xi Yang、Liang Hu、Zhou Huang、Shuang Wei、Longrui Yang、Yanchi Chen、Lanpeng Jia
- **机构：** Changhong Intelligent Robot；University of Electronic Science and Technology of China（电子科技大学，UESTC）。arXiv HTML 作者栏未能稳定对应每位作者的机构标记，因此不推断个人归属。
- **arXiv：** [2608.14160](https://arxiv.org/abs/2608.14160)
- **核对版本：** v2，2026-09-17 修订；[v2 摘要](https://arxiv.org/abs/2608.14160v2) · [v2 HTML 正文](https://arxiv.org/html/2608.14160v2) · [v2 PDF](https://arxiv.org/pdf/2608.14160v2)
- **项目页 / 官方代码 / 权重：** 论文及可核实的一手入口未提供可确认链接；不把 NavDP、π³ 或其他相关工作仓库误列为 OccPlanner 源码。
- **数据：** 论文使用 InternData-N1 与 InternScenes，但 v2 页面没有提供可直接核对的数据下载地址或许可信息。
- **许可边界：** arXiv 页面显示论文内容适用 arXiv perpetual non-exclusive license；这不是代码许可，不能据此推断实现或数据已开源。

## 核心内容

OccPlanner 将图像中的 PixelGoal 与 RGB-D 历史观测共同用于连续轨迹生成。模型学习两种互补表示：目标在机器人坐标系中的度量位置/朝向，以及规划相关的局部三维占据；二者共同条件化扩散轨迹模块。

L3ROcc 是论文提出的几何监督生成流程，不是与 OccPlanner 分开的独立项目：它从单目 RGB 导航视频出发，用 π³ 做多帧几何重建和相机位姿估计，再进行尺度恢复、机器人坐标系对齐、体素化及射线可见性推理，生成轨迹与局部占据监督。体素区分为可见占据、已观测自由和未观测区域。

## 关键实验

- 仿真：NVIDIA Isaac Sim 中以 Clearpath Dingo 评测 60 个未见场景（Home 20、Commercial 20、Cluttered Easy 10、Cluttered Hard 10）。3–5 m / 5–8 m 两个距离段过滤无效输出后，分别保留 2,437 和 2,672 个回合。
- 5–8 m 成功率（%）：

| 方法 | Home | Commercial | Cluttered Easy | Cluttered Hard |
|------|-----:|-----------:|---------------:|---------------:|
| NavDP-PixelGoal | 9.46 | 8.62 | 18.45 | 20.12 |
| OccPlanner | 47.83 | 45.81 | 94.78 | 91.44 |
| iPlanner（PointGoal） | 49.70 | 52.16 | 96.11 | 95.88 |

NavDP-PixelGoal 是作者对已发表 PointGoal NavDP 的 PixelGoal 适配。论文报告 OccPlanner 在 8 个场景—距离组合中均超过 PixelGoal 对照；远程分段与最强 PointGoal 参考的总体差距在 3.50 个百分点以内。PointGoal 直接获得度量目标，不能与 PixelGoal 输入条件混为一谈。

- Go2 实机闭环：Unitree Go2 + Orbbec Gemini 336L RGB-D 相机；SAM 3 以 1 Hz 更新目标，Dynamic-VINS 以 20 Hz 估计自运动，MPC 跟踪路点并输出速度命令。每种训练设置各 20 次试验：

| 模型设置 | 成功 | 碰撞 |
|----------|-----:|-----:|
| 仿真训练（zero-shot） | 11/20（55%） | 12/20（60%） |
| 用 829 个实机样本微调 | 16/20（80%） | 5/20（25%） |

成功率与碰撞率独立统计。论文报告 120M 参数、8 帧 224×224 RGB-D、24 个轨迹增量、10 次扩散去噪；单次推理 0.104 s，测试硬件为 RTX 4090。该耗时不是板载算力或控制频率指标。
