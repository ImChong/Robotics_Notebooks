# VLFM 论文通读（零样本目标导航视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/zero-shot-object-navigation.md`）

- **标题：** VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation
- **作者：** Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, Bernadette Bucher
- **类型：** paper / navigation / zero-shot-objectnav
- **原文：** <https://arxiv.org/abs/2312.03275>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 用 BLIP-2 对当前 RGB 与目标文本算余弦相似度，投影成带置信度的俯视价值图，用它给占据图前沿打分；检测到目标后用预训练 PointNav 策略走到目标，在 Gibson / HM3D / MP3D 上刷新零样本 ObjectNav 的 SPL。
- **核对范围：** 通读摘要、§I–§VII（问题设定、前沿与价值图、检测、PointNav、基准、消融、真机、结论与局限）及参考文献列表；该论文无独立附录。未阅读其引用的 ESC、CoW、ZSON、PIRLNav 等原文，相关对比数字仅取自本文表 I。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **任务设定（§III）：** 只有第一视角 RGB-D 与里程计；动作为前进 0.25 m、左右转 30°、抬头低头 30°、STOP；在 500 步内于任一目标实例 1 m 内 STOP 才算成功。
- **三阶段（§IV）：** 初始化（原地转一圈建图）、探索（持续更新前沿图与价值图，选价值最高的前沿）、目标导航（检测到目标后走向其最近点并 STOP）。无前沿且未检测到目标则 STOP 判失败。
- **前沿生成（§IV-A）：** 深度转点云，滤掉过低过高点，投影到 2D 栅格；已探索与未探索区域的每条边界取中点作为前沿航点。
- **价值图（§IV-B）：** 用 BLIP-2 对当前 RGB 与提示 "Seems like there is a <target> ahead." 直接算余弦相似度（不先把图像转成文字）；投影到俯视栅格（相机视锥形并排除被遮挡区域）。另有置信度通道，置信度 cos²(θ/(θ_fov/2)·π/2)：光轴为 1、视野左右边缘为 0；重复观测时按置信度加权平均更新价值，置信度更新偏向较高者。
- **目标检测（§IV-C）：** COCO 类别用 YOLOv7，其余用 Grounding-DINO；检出后用 Mobile-SAM 取轮廓，结合深度求离机器人最近的目标点作为目标航点。
- **航点导航（§IV-D）：** PointNav 策略用 VER 在 HM3D 训练集上训练（4 GPU、每卡 64 worker、25 亿步、约 7 天），只吃深度和相对目标距离 / 朝向，不用 RGB；可换成任何能走向视觉可见航点的方法，真机不用该策略。
- **基准（§V–VI，表 I）：** HM3D：SPL 30.4 / SR 52.5；MP3D：17.5 / 36.4；Gibson：52.2 / 84.0。零样本方法上相对此前最好成绩，Gibson 比 SemUtil 高 11.7 SPL / 14.7 SR，HM3D 比 ESC 高 8.1 SPL / 13.3 SR，MP3D 比 ESC 高 3.3 SPL / 7.7 SR。HM3D 上 SR 比有监督的 PIRLNav 低 11.6，但 SPL 高 3.3。
- **局限（§VI-A、§VII）：** 只支持单楼层（模拟器里的里程计没有 z 坐标），HM3D 与 MP3D 分别有 14.6% 与 9.6% 的 episode 因需要上下楼梯而失败；假设目标在默认相机高度下容易被看到（不会开抽屉等）；价值图只含与当前目标有关的语义，无法跨目标复用或直接用于 VLN。
- **消融（§VI-B，表 II）：** 价值更新方式：置信度加权平均优于直接替换与不加权平均（HM3D SPL 30.4 / 30.0 / 26.5，MP3D 17.5 / 17.1 / 16.6，Gibson 52.2 / 50.9 / 48.0，依次为加权平均 / 不加权平均 / 替换）。
- **真机（§VI-C）：** Boston Dynamics Spot 上用其 API 做航点导航；用 ZoeDepth 估计深度来近似目标航点；BLIP-2、Grounding-DINO、MobileSAM、ZoeDepth 全在一台带 RTX 4090 笔记本 GPU（16 GB 显存）的笔记本上实时运行；真机部分为演示，文中未给出定量成功率。
- **与同类对比（§II）：** CoW 去最近前沿直到检测到目标；ESC、LGX 用 LLM 评估由检测结果转成的文本，SemUtil 用 BERT 嵌入类别名；VLFM 的区别是直接从 RGB 取图文相似度，避免视觉先转文字的瓶颈与 LLM 算力需求。

## 对 wiki 的映射

- [`wiki/tasks/zero-shot-object-navigation.md`](../../wiki/tasks/zero-shot-object-navigation.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2312.03275>
