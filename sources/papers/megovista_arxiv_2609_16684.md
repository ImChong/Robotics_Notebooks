# MEgoVista: Multi-view Ego-aware Motion Estimation for Metric 4D Hands and Head in the Wild

> 来源归档；入库及开放状态核查：2026-10-12

- **类型：** paper / metric reconstruction / egocentric human data
- **论文：** <https://arxiv.org/abs/2609.16684>
- **全文：** <https://arxiv.org/html/2609.16684v1>
- **PDF：** <https://arxiv.org/pdf/2609.16684>
- **首次提交：** 2026-09-15
- **作者：** Jiangong Xiao、Zhihao Zhang、Yifei Dong、Chao Ma、Zhouyi Jin、Zhiwen Hou、Li Liu、Weihuang Chen、Hongbin Sun、Maoqing Yao
- **机构：** 西北工业大学、 西安交通大学、Maniformer
- **硬件官网：** <https://www.maniformer.ai/en/mego> — [核查归档](../sites/megovista-mego-view.md)
- **实现开放状态：** 未发现 MEgoVista 专属公开训练/重建代码、权重或评测数据下载入口；硬件产品页不是算法源码仓。
- **论文许可：** arXiv v1 标注 CC BY-SA 4.0；不等于代码、数据或 MANO 许可。
- **一句话说明：** 从同步三鱼眼与惯性录制生成公制双手和头部轨迹，用已标定双目基线确定尺度，以独立光学动捕审核精度。

## 原文摘录要点与知识映射

### 1. 硬件与可观测性（§3.1、§3.3）

MEgo View 提供 60 fps 多相机录制及 500 Hz IMU。头部 SfM 将刚性阵列视为广义相机，固定 Kannala–Brandt 标定；已知左右相机基线进入初始化，IMU 用于关键帧候选及视觉完成后的重力对齐，不进入联合视觉惯性 BA。

**对 wiki 的映射：** [MEgoVista 主节点](../../wiki/entities/paper-megovista.md)；尺度测量与事后相似变换对齐的区别。

### 2. 手部重建（§3.2）

检测同时判断位置、左右手和佩戴者归属，防止旁人手混入。MANO 初值与两个 MID×侧相机深度经质量筛选进入完整 episode 双手联合优化；约束包括重投影、腕深度、姿态先验、时间连续性和解剖软障碍。

**对 wiki 的映射：** [动作重定向管线](../../wiki/concepts/motion-retargeting-pipeline.md) — 作为人示范标注上游，不直接提供机器人关节目标。

### 3. 真值与指标（§4.1–4.4、Tables 1–3）

独立参考为 26 相机、120 Hz Chingmu，PTP 同步。头部误差使用固定尺度 hand-eye 对齐；PA-MPJPE-p 使用 Procrustes 相似对齐，剥离手尺度和腕朝向，所以只衡量手指 articulation。校准、验证和评测录制分离，棋盘验证不能证明厂商手骨架求解器没有慢漂。

**对 wiki 的映射：** [MEgoVista 主节点](../../wiki/entities/paper-megovista.md) — 指标口径和参考误差的边界。

### 4. 结果与限制（§4.3–4.5）

稳态头 ATE 为 Session 1 的 2.83 mm、Session 2 的 0.65 mm；ORB-SLAM3 对照是离线精炼结果，第一组稳态为 2.12 mm，不能概括为本方法全面获胜。手 PA-MPJPE-p 为 4.29 mm，jitter 为 1.11 mm/frame²；约 40 秒录制的 motion tracking 耗时约 206 秒，不是实时输出吞吐。野外奶茶制作、美甲和动物清洗没有动捕真值，仅展示定性投影。

原文 §3.2 描述完整序列优化，但 §4.4 的平滑性段落又写无 test-time optimization；边界未解释清楚。入库保留此歧义，不擅自推断实时或免优化。

**对 wiki 的映射：** [MEgoVista 主节点](../../wiki/entities/paper-megovista.md) — 工程验收与结论。
