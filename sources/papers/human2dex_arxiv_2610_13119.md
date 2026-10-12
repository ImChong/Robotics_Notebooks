# Human2Dex 论文来源归档

## 元数据

- **题目：** Human2Dex: Dexterous Robot Learning from Wrist-Centric Human Demonstrations
- **作者：** Jiachen Zhang, Yuanyuan Li, Shaqi Luo（通讯）
- **单位：** Beijing Academy of Artificial Intelligence（北京智源人工智能研究院）
- **预印本：** https://arxiv.org/abs/2610.13119
- **HTML：** https://arxiv.org/html/2610.13119v1
- **PDF：** https://arxiv.org/pdf/2610.13119
- **版本：** v1，2026-10-09，cs.RO
- **项目页：** https://human2dex.pages.dev/（归档：[human2dex-pages-dev.md](../sites/human2dex-pages-dev.md)）
- **代码：** https://github.com/zgates2/Human2Dex_（归档：[human2dex.md](../repos/human2dex.md)）
- **入库日期：** 2026-10-12

## 论文要解决的问题

不同灵巧手每换一只就要重新采集示范。作者问：同一批自然人手示范能否成为多只灵巧手共用的数据源，同时仍允许各手各自学习与本体相关的动作。

## 方法与训练信息

1. **腕部视角采集：** 人戴腕部相机徒手演示；人手 3D 结构来自 PICO 追踪与腕部 RGB 手部模型（DINO 编码器 + 3D MANO 分支 + 2D 关键点分支）。两只目标手（Linker O6、Wuji）也装标定过的腕部鱼眼相机。
2. **局部手部外观适配：** 外观随机化只作用于手部掩码内，物体与背景不变。
3. **统一 21 点手部表示：** 人与机器人观测共享同一关键点拓扑、配色与坐标约定；机器人关键点由正运动学加相机投影（固定手眼标定）得到。
4. **融合与重定向：** PICO 与腕部 RGB 估计按置信度加权插值，再经自适应卡尔曼平滑；Linker O6 用 DexPilot 风格目标（15 条几何向量，解 6 个主动关节）；Wuji 用扩展目标，SLSQP 求解 20 DoF。动作维度为 9+d_r（d_O6=6，d_Wuji=20）。
5. **GPIR（Grasp-Pocket Interaction Representation）：** Functional Grasp Pocket 取图像空间拇指与食指指尖中点；指尖轨迹经因果三帧中值滤波；每个被跟踪物体给出相对 Pocket 的质心偏移、对数面积比、跟踪置信度与记忆有效标志。
6. **策略：** DINOv2 ViT-B/14（带 4 个 register）图像编码器 + 1-D U-Net（通道 256/512/1024）扩散头，预测 16 步动作；AdamW lr 3e-4，batch 32，一帧历史，50 训练步 / 16 DDIM 推理步，5% episode 留作验证；部署 16 Hz，每个动作块执行前 8 步；GPIR 辅助损失为训练期使用的 masked Smooth-L1。
7. **数据规模：** 1,899 条质检后基础 episode、556,624 帧、6 个任务；每条基础 episode 渲染 3 个外观变体。每任务示范数原文未报告。每只手单独训练策略（动作标签、归一化、检查点均按手区分）。

## 论文报告的结果

六项真机任务（Sponge、Croissant、Cup、Bottle、Egg Carton、Open Drawer），每任务每手 20 次：

| 配置 | Wuji | Linker O6 | 合计 |
|------|------|-----------|------|
| Human-Raw | 19/120（15.8%） | 44/120（36.7%） | 63/240（26.3%） |
| Human-Visual | 65/120 | 79/120 | 144/240（60.0%） |
| Human-Structured | 74/120 | 89/120 | 163/240（67.9%） |
| Human2Dex（完整） | 97/120（80.8%） | 109/120（90.8%） | 206/240（85.8%） |

- Human2Dex 的 Wilson 95% 区间：Wuji 72.9–86.9%，O6 84.3–94.8%。
- 腕部 vs 头部视角（O6-Croissant、Human-Visual 配置，各 20 次）：腕部 6/20（首次抓取 4/20），头部 0/20；单手单任务诊断。
- 参考系对比（O6）：Croissant Human-Absolute 17/20 对 Human2Dex 18/20；Cup 9/20 对 16/20。
- 增益主要在抓取中心任务（Sponge/Croissant/Cup/Bottle）；Egg Carton 与 Open Drawer 增益小或没有，这两项依赖局部区域（盖沿、抽屉把手）。
- 论文自述：逐级配置比较不能分离单个组件的因果效应。

## 复现与解释边界

- 仅两只标定好的拟人手、六项任务，各自独立策略；任意形态或共享动作空间的迁移未测。
- GPIR 需要指定任务物体与可靠的图像空间跟踪，整体物体表示无法指出局部接触区域。
- 没有机器人示范基线，人机数据样本效率对比留作未来工作。
- HTML 版本抽取完整，附录细节本页未逐节核对。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-human2dex-wrist-centric-dexterous.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- <https://arxiv.org/abs/2610.13119>、<https://arxiv.org/html/2610.13119v1>
- 项目页：<https://human2dex.pages.dev/>；代码：<https://github.com/zgates2/Human2Dex_>
