# ReGATE（arXiv:2611.02871）

> 来源归档（paper）

- **标题：** ReGATE: Recurrent Gated Visual Fusion for Reactive Humanoid-Object Interaction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.02871>（v1，2026-11-02，cs.RO）
- **HTML 全文：** <https://arxiv.org/html/2611.02871v1>
- **PDF：** <https://arxiv.org/pdf/2611.02871>
- **作者 / 机构：** Hao Zhu、Jorge Peña-Queralta；Centre for Artificial Intelligence, Zürich University of Applied Sciences（ZHAW，苏黎世应用科技大学人工智能中心），Winterthur, Switzerland
- **代码：** 论文正文未给出代码或项目页链接（入库日 2026-11-09 未核查到公开仓库）
- **入库日期：** 2026-11-09
- **核对范围：** 读取 arXiv HTML v1 全文：摘要、引言、相关工作、方法 III-A 至 III-D、实验 IV-A/IV-B、结论、致谢与参考文献列表；图只读了图注（未逐帧看图像），表 I-III 数值已与正文核对。该版本无附录、无补充材料。未读被引论文（VAIC、VisualMimic 等）原文。
- **通读：** 已完成
- **一句话说明：** 面向 Unitree G1 的人形物体交互全身控制器：teacher-student 框架下，循环门控深度-本体融合适配器估计物体几何与接触，在深度被遮挡或中断时仍保持交互。

## 核心摘录

1. **任务与接口（III-A）：** 策略输入本体观测、egocentric 深度图（1×H×W）与因果命令 u=[vx, vy, ωz, c]（c 为期望接触 0/1），输出 23 维关节位置目标，由 PD 跟踪；不使用外部物体位姿估计器。
2. **适配器（III-B）：** 深度图拼接归一化图像坐标后经 CNN，再经空间 ConvGRU 得到空间深度 token；本体+命令的 MLP 特征作为 query，8 头交叉注意力取深度证据；头级门 g_head 与标量全局门 g_global 融合：z_fuse = LN(z_p + g_global * (g_head · A))；g_global=0 时退化为纯本体残差路径。辅助头预测物体状态、带位姿点云、末端接触，并由其条件化适配 GRU 输出交互隐变量。
3. **训练（III-C）：** 阶段 I 用 PPO 训练特权教师并以零深度特征预热学生适配模块；阶段 II 蒸馏学生（适配损失 Eq.4 + 动作均值匹配），学生演员再由 PPO 微调。深度课程：先干净深度并强制门全开（避免门过早关闭），再逐步放开门并引入结构化遮挡与整帧黑屏。阶段感知随机状态初始化把示范分为 approach / pre-contact / interact / pre-release / release 五阶段。
4. **实验设置（IV-A）：** Isaac Sim 训练，教师 4096 并行环境，学生 512，一块 RTX 5090；策略 50 Hz，深度 64×36 @15 Hz；真机 Unitree G1（29 自由度，策略控制 23 关节，6 个腕关节固定为零位），头戴 ZED 2i；评测在 MuJoCo 与真机上做箱子拾取。
5. **结果（IV-B）：** 真机命令序列 10/10 成功（Q1）；MuJoCo 静态箱子 281/330（85.2%），VisualMimic 同协议 160/330（48.5%）（表 I、II）；移动箱子 378/440（85.9%）（表 III）；真机展示相机遮挡后仍抓取箱子（图 7，定性）。
6. **局限（论文自述，IV-A4）：** 明确不评测接近物体与接触触发决策，设想由 VLM 或其他高层控制器负责。

## 与周报说法的核对

- 机构：与论文首页一致（ZHAW 人工智能中心）。
- 创新点中「通道级和全局门控」：论文是「head-wise（注意力头级）和 global 门」，不是通道级。
- 「相机短暂遮挡后仍能继续完成交互」：论文真机实验是在初次观测后、接触命令前遮挡相机并成功取箱（图 7，仅定性展示，未给次数）；仿真黑屏协议（Q3）无定量表。

## 对 wiki 的映射

- [paper-regate](../../wiki/entities/paper-regate.md)
