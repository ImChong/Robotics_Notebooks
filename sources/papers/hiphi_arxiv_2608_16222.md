# HiPHI: A Large-Scale Benchmark for High-Precision Human Motion and Object-Interaction

> 来源归档（ingest）；以论文 v2、官方项目页与发布文档为准。

- **类型：** paper / dataset / humanoid / human-object-interaction
- **作者：** Jiahao Ji、Ji Ma、Runhan Zhang 等
- **机构：** 诺亦腾机器人；新加坡国立大学；清华大学深圳国际研究生院；香港科技大学；香港大学
- **来源日期：** 2026-08-17（v1）；2026-09-08（v2）；CoRL 2026 接收
- **入库日期：** 2026-10-02
- **论文：** <https://arxiv.org/abs/2608.16222>
- **全文：** <https://arxiv.org/html/2608.16222v2>
- **项目页：** <https://noitom-robotics.github.io/hiphi/>（[归档](../sites/hiphi.md)）
- **代码：** <https://github.com/noitom-robotics/hiphi>（[归档](../repos/hiphi.md)）
- **数据集：** <https://huggingface.co/datasets/noitomrobotics/HiPHI>
- **在线 Viewer：** <https://hiphi-viewer.modalitynet.com/>
- **一句话说明：** 用 FrameNet 组织动作采集，提供人体运动及同步物体几何，连接数据覆盖、质量与人形跟踪评测。

## 核心论文摘录（MVP）

### 1) 覆盖设计：语义种子不是任务数量

- **链接：** 论文 §3；项目页 Frame–LU construction。
- **摘录要点：** Frame 表示事件类型，LU 表示词在该事件中的具体词义；22 个 Frame、214 个 Frame–LU 用于设计采集脚本，再按速度、方向、姿态、幅度和交互条件扩展。目的是补全控制相关运动空间，而非堆相似任务名称。
- **对 wiki 的映射：** [HiPHI 核心原理](../../wiki/entities/paper-hiphi.md)；[具身数据纵深](../../roadmap/depth-embodied-data.md)。

### 2) 数据与物体：发布规模包含镜像增强

- **链接：** 论文 §4、附录 B；官方 README 的 Overview / HOI Format。
- **摘录要点：** 发布 617.5 h、200.1M 帧，原始采集约 308.7 h；包含 body-only 371.8 h 和 HOI 245.7 h。90 Hz、55 关节 BVH、132 名表演者；HOI 同步 CSV 物体位姿与 OBJ 网格。人体/网格厘米、物体平移米，转换时不能共用未经检查的缩放。
- **对 wiki 的映射：** [HiPHI 数据与工程实践](../../wiki/entities/paper-hiphi.md)；[人形参考数据集选型](../../wiki/comparisons/humanoid-reference-motion-datasets.md)。

### 3) 评测：覆盖与跟踪分别检验

- **链接：** 论文 §5、附录 G/H；项目页 Benchmark / Humanoid learning。
- **摘录要点：** 统一表征后做覆盖分析，另检查平滑度、地面接触与物体几何；相同数据预算和跟踪设置下比较数据源。未镜像训练量从 3 h 扩至 300 h，跨数据集 MPJPE 持续下降；G1 展示移动、姿态变换与物体交互。HOI 并非每个子指标都最优，几何一致性不等于力学接触证明。
- **对 wiki 的映射：** [HiPHI 实验与结论](../../wiki/entities/paper-hiphi.md)。

### 4) 发布核查：可浏览不等于完整训练可复现

- **链接：** 官方仓库 README、viewer/README.md；Hugging Face 数据卡。
- **摘录要点：** GitHub 已发布本地 Viewer 与格式文档；本次检查未找到论文训练/真机部署入口或策略权重。数据已发布但需 HF 申请访问，采用 ModalityNet Open Research License v1.0；Viewer 的 Apache-2.0 许可另算。
- **对 wiki 的映射：** [HiPHI 开放边界](../../wiki/entities/paper-hiphi.md)；[源码归档](../repos/hiphi.md)。

## 当前提炼状态

- [x] 论文、项目页、仓库和数据卡核对
- [x] 论文实体页、数据集比较与两条纵深路线接入
- [x] 标注镜像统计、单位同步与数据访问边界
