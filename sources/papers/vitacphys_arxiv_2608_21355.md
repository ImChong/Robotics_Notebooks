# ViTacPhys（arXiv:2608.21355）

> 来源归档（ingest）

- **标题：** ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations
- **类型：** paper / visual-tactile / physical-property / adaptive-grasping / imitation-learning
- **arXiv：** <https://arxiv.org/abs/2608.21355>
- **项目页：** <https://vitacphys.github.io/ViTacPhys/>
- **机构：** 小米机器人实验室（Xiaomi Robotics）— Yiwen Liu、Yujun Zhu、Kui Jia、Zhao Liao、Yangwei You、Shuaijun Wang（通讯）
- **状态：** 2026-08 arXiv 预印本，同行评审前
- **入库日期：** 2026-08-24
- **一句话说明：** 从人体视触觉抓取示范在线估计质量/刚度/摩擦，经人→机迁移条件化 ACT 灵巧手策略，实现属性自适应抓取。

## 开源状态（步骤 2.5，2026-08-24）

| 资源 | 状态 |
|------|------|
| arXiv HTML/PDF | **已发布** |
| 项目页 Code | **Coming Soon** — 截至入库日无 URL |
| 项目页 Dataset | **Coming Soon** — 截至入库日无 URL |
| 结论 | **宣称将开源 / 待发布** |

## 核心论文摘录

### 1) 问题与数据集（Abstract / §III）

- **核心贡献：** 显式预测操作相关物理属性（质量三档、连续刚度、摩擦三档），而非仅靠隐式视觉策略；构建 **60** 物体、**1800** 条 1 s@30 Hz 同步人体视触觉抓取示范，含精密秤/力–位移/斜面摩擦标注协议。
- **采集：** 可穿戴手套（三指压感图 180×40 + 腕 RGB 160×90×3 + 动捕）；垂直提起与侧向摇晃两协议，每物体每协议 15 试次。
- **对 wiki 的映射：**
  - [ViTacPhys 论文实体](../../wiki/entities/paper-vitacphys.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)

### 2) ViTacPhys 预测器（§IV）

- **时序编码：** ResNet-18 内容流 + Farnebäck 光流运动流 → GRU；双向 cross-attention 融合视触觉。
- **VLM 语义先验：** 接触前 **5** 帧 RGB 经 GPT-5.4 生成文本假设 → 冻结 Sentence-BERT/BERT 编码；门控 cross-attention 注入，不暴露接触后形变。
- **损失：** 质量/摩擦用 **有序回归**；刚度 MSE；**GradNorm** 平衡三任务。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 3) 人→机迁移与下游策略（§IV-F / §VI）

- **迁移：** 人机同型传感器；**Seedance 2.0** 机器人风格 RGB 增广 + 匹配动作人体示范 + 有限遥操作；自人体 ID 预训练微调。
- **策略：** 预测属性嵌入为 **4×512 token**（接触标志 + 质量/刚度/摩擦索引）条件化 **ACT**；接触后滚动 30 帧队列 + 时序投票稳定预测；Jetson Orin **9 ms + 10 ms** @30 Hz。
- **对 wiki 的映射：**
  - [Action Chunking](../../wiki/methods/action-chunking.md)
  - [ViTacFormer](../../wiki/entities/paper-sa-2506-15953-vitacformer-learning-cross-modal-representation.md)（基线）

### 4) 实验（§V–VI）

- **属性预测（held-out object）：** 质量 Acc **87.5%**、摩擦 **97.5%**、刚度 MAPE **9.08%**。
- **真机抓取（ViTacPhys Pred.）：** ID 总成功率 **95.0%**、OOD **83.4%**；相对 ACT clean-success **+12.5 / +38.9 pp**（ID/OOD）；OOD 力剖面更接近遥操作。
- **平台：** 7-DoF 臂 + 6-DoF 连杆灵巧手；40 ID + 6 对视觉相似 OOD；每物体 10 条遥操作示范。
- **对 wiki 的映射：**
  - [Grasp Pose Estimation](../../wiki/methods/grasp-pose-estimation.md)
  - [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)

### 5) 局限

- 触觉主要为法向压感，质量/摩擦只能分档；刚度为操作级测量非材料本征；**60** 物体、**1** 名采集者；OOD 仅 6 对物体；VLM 先验一次性约 **10 s** 延迟。
