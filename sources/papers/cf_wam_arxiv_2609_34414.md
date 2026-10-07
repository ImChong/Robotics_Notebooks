# CF-WAM 论文来源归档

## 元数据

- **题目：** From World Models to World Action Models: Rethinking Next-State Prediction
- **方法：** Confluent Foresight World Action Model（CF-WAM）
- **作者：** Tingyu Yuan, Ziming Ji, Biaoliang Guan, Wen Ye, Wenrui Tian, Zhaopeng Gu, Feihong Zhang, Xu Yang, Yan Huang, Zhaowen Li, Chaoyang Zhao, Jinqiao Wang
- **单位：** Institute of Automation, Chinese Academy of Sciences；University of Chinese Academy of Sciences；Beijing University of Posts and Telecommunications；Xi’an Jiaotong University；Wuhan University；Tsinghua University；Yinwang Intelligent Technology Co. Ltd.
- **预印本：** https://arxiv.org/abs/2609.34414
- **HTML：** https://arxiv.org/html/2609.34414v1
- **PDF：** https://arxiv.org/pdf/2609.34414
- **版本：** v1，2026-09-28，cs.RO
- **代码与数据：** 论文写明将在录用后发布；arXiv 页面当前没有链接官方代码仓库。

## 论文要解决的问题

世界模型通常把未来状态预先固定成 RGB 视频或某个潜特征；世界–动作模型（WAM）还要用这个未来预测支持动作生成。固定表示会把某一种表征的偏置带进策略：RGB 保存丰富外观但包含大量与动作无关的细节；语义表示突出物体与任务结构、却弱化空间关系；几何表示保留三维布局、但缺少对象身份与交互进程；交互分割突出操作者与物体接触变化、但缺少全局场景。

CF-WAM 将这些表示作为同一动作条件未来的互补投影，而不是同时建立四套预测分支。每个训练样本动态抽取一种未来表示，使来自不同投影的动作相关约束在训练更新中逐步累积。

## 方法与训练信息

1. 将未来表示统一为视频形式，并使用共享的视频 VAE；目标类型包括 RGB、DINOv3-PCA 语义视频、深度视频、手–物体交互分割视频。
2. 每个样本只抽取一个表示，并把表示提示附加到任务指令中。当前 RGB 帧仍作为各未来视频共同的起始条件。
3. WAM 由 Mixture-of-Transformers 构成：预训练 Wan2.2-TI2V-5B 视频专家预测未来，ActionDiT 动作专家预测连续控制轨迹；两路参数独立，在每层通过非对称混合注意力交换信息。
4. 视频和动作支路分别使用 flow matching。动作专家输出 16 步、每步 47 维的动作块。
5. 人类与机器人数据不要求逐帧或逐任务配对。EgoDex 第一视角数据提供双腕末端执行器运动；GR-1 机器人数据提供相同末端执行器空间及机器人专属关节动作。人类样本没有的 29 维关节动作由来源掩码排除，不用零值监督。
6. 机器人样本来自 RoboCasa-GR1 的桌面任务数据：24 项任务，每项 1,000 个 episode。人类部分使用 EgoDex 的 pick-and-place 子集。论文将人类数据从 30 Hz 对齐到 20 Hz 控制采样。

视频专家以 Wan2.2-TI2V-5B 初始化；动作专家约 1.021B 参数，两个专家均有 30 层，合计约 6.021B 可训练参数（不含冻结 VAE 和文本编码器）。RoboCasa-GR1 主实验使用 4 台节点、每台 8 张 NVIDIA B200、全局 batch 32、训练 100K 步；LIBERO 训练使用单节点 B200、80K 步。

## 论文报告的结果

- **RoboCasa-GR1：** 24 项双臂/灵巧手桌面任务，每项评测 50 个闭环 rollout；CF-WAM 平均成功率 82.50%，WALA 为 75.17%，差 7.33 个百分点。CF-WAM 在 24 项任务中的 15 项排名第一。
- **LIBERO-Plus：** 在 LIBERO 训练后不对测试环境做适配，CF-WAM 平均 82.65%；FoMoVLA 为 80.50%，GaussianWAM 为 77.30%。
- **真机六任务：** Chemistry、Folding、Organization、Fruit Sorting、Insertion、Wiping，每项每个设置 50 次闭环试验。Interaction 单一推理投影平均为 84.00%；按每项任务选择最佳投影的 oracle 汇总为 87.00%，不能当成无需选择的单一部署结果。
- **真实场景 OOD：** 五种场景变化下 Interaction 投影平均 79.20%。Visual 推理加入人类数据后为 75.60%；移除人类数据后为 45.60%，显示该设置中人类经验贡献明显。
- **动态与静态多表示消融：** 同一模型下动态采样四种表示达到 79.00%，最佳单表示为 76.25%；每步同时预测四种表示的 Static Multi-State 为 68.75%。该消融与完整人–机器人主结果采用不同设置，数值不应混作一项基准。

## 复现与解释边界

- arXiv v1 是预印本，论文承诺在录用后发布代码与数据；当前没有可验证的官方仓库或模型权重链接。
- 多项基准结果依赖 Wan2.2-TI2V-5B 初始化及 B200 训练资源；论文报告的性能不是轻量级本地复现承诺。
- 真机结果覆盖六个任务并采用闭环试验，但不能外推为所有机器人形态或开放环境的通用能力。
- 87.00% 是根据每个任务结果挑选投影的 oracle 统计；固定 Interaction 投影结果为 84.00%。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-cf-wam-dynamic-next-state-prediction.md
- 概念入口：wiki/concepts/world-action-models.md
- 方法入口：wiki/methods/generative-world-models.md
- 任务入口：wiki/tasks/bimanual-manipulation.md

## 一手来源

- arXiv 摘要与版本：<https://arxiv.org/abs/2609.34414>
- arXiv HTML 全文：<https://arxiv.org/html/2609.34414v1>
- arXiv PDF：<https://arxiv.org/pdf/2609.34414>
