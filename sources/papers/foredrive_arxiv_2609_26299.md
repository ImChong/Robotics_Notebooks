# ForeDrive（规划相关潜在世界模型）

> 来源归档（ingest）

- **标题：** ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.26299>
- **HTML 全文：** <https://arxiv.org/html/2609.26299v2>
- **PDF：** <https://arxiv.org/pdf/2609.26299>
- **作者：** Sinuo Wang, Zichong Gu, Yuhan Huang, Wenxin Wen, Xun Yang, Yiqing Zhang, Xingyu Zhang, Ningyu Che, Jie Ling, Qiankun Yu, Wei Liu, Jing Xu, Xinggang Wang
- **机构：** Huazhong University of Science and Technology；Shanghai Zaofu Intelligent Technology Co., Ltd.；Tongji University
- **版本：** arXiv v2，2026-09-23
- **项目页 / 代码：** 截至 2026-10-05，未找到官方项目主页或代码仓库
- **模型 / 数据：** 未找到可下载的官方 checkpoint 或独立数据发布；论文使用 NAVSIM 数据与评测协议
- **入库日期：** 2026-10-05
- **一句话说明：** 用 JEPA 式多时域潜变量预测为扩散轨迹规划提供未来线索，同时通过非对称梯度路由让规划目标塑造共享视觉编码器、避免规划梯度直接改写未来预测器。

## 核心摘录（MVP）

### 1) 从未来可预测性转向规划相关表征

- **摘录要点：** 论文指出，未来预测得准不等于对规划有用。ForeDrive 用共享在线视觉编码器与 EMA 目标编码器构成 JEPA 式世界模型，针对多个未来时域预测视觉潜变量和自车状态；未来图像仅用于训练目标，推理时不可见。
- **对 wiki 的映射：**
  - [ForeDrive 论文实体页](../../wiki/entities/paper-foredrive.md)
  - [生成式世界模型](../../wiki/methods/generative-world-models.md)

### 2) 非对称优化与规划接口

- **摘录要点：** 规划损失可更新共享在线编码器；预测器只由潜变量/状态预测损失训练，送入规划器的未来表征采用 stop-gradient。规划器使用当前观测为主干，通过门控视觉融合、未来状态注入和 Trajectory-Adaptive Bias（TAB）消费预测未来。TAB 将候选轨迹投影到前视相机，并在去噪中偏置路径相关图像 token 的注意力。
- **对 wiki 的映射：**
  - [ForeDrive 论文实体页](../../wiki/entities/paper-foredrive.md)
  - [世界模型功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)

### 3) NAVSIM 结果与消融读法

- **摘录要点：** 主配置在 NAVSIM v1 报告 89.9 PDMS，在 NAVSIM v2 报告 90.0 one-stage EPDMS；两项属于不同版本、不同指标，不能直接横向比较。v1 的 ViT-L 容量上界为 90.4 PDMS。匹配的 current-only Base 为 88.9 PDMS，完整 WM+TAB 为 89.9；论文消融显示，单加 WM 或 TAB 分别提升 0.7 / 0.6 分，联合提升 1.0 分。
- **对 wiki 的映射：**
  - [ForeDrive 论文实体页](../../wiki/entities/paper-foredrive.md)
  - [DiffusionDrive](../../wiki/entities/paper-diffusiondrive.md)

### 4) 推理成本与复现边界

- **摘录要点：** 论文报告单张 NVIDIA H20 上默认模型推理 58.2 ms/frame（17.2 FPS），世界模型相对匹配 Base 增加约 26 ms；评测计时为纯 FP32 前向，不含裁剪/缩放和数据加载。补充材料给出 PyTorch 2.4、Python 3.9、CUDA 12.1，以及 32 张 H20 训练配置。当前未找到官方代码、权重、项目页或独立数据包，因此这些是论文报告的实验条件，不构成可运行复现流程。
- **对 wiki 的映射：**
  - [ForeDrive 论文实体页](../../wiki/entities/paper-foredrive.md)

## 来源状态

- 论文主体与补充材料：arXiv v2 HTML/PDF。
- 未找到官方项目页、代码仓库、可下载权重或单独数据集链接（截至 2026-10-05）。
- 论文报告 NAVSIM 与 nuScenes zero-shot 评测；这不等于真车部署验证。
