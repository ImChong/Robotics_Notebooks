# DirtyMoCap: Robust Motion Capture from Unconstrained Markers

> 来源归档（ingest）；论文细节以 arXiv v1 为准，代码/数据开放状态另按官方项目页与仓库核查。

- **类型：** paper / motion-capture / human-motion-reconstruction
- **作者：** Long Wang、Shuting Zhao、Shen Yan、Siyuan Yu、Xiaoben Li、Zeyu Cai、Yumeng Hou、Yuliang Xiu
- **机构：** 浙江大学；西湖大学；复旦大学；国防科技大学；南京大学；新加坡国立大学
- **来源日期：** 2026-09-17（arXiv v1）；官方项目页列 SIGGRAPH Asia 2026
- **入库日期：** 2026-10-14
- **论文：** <https://arxiv.org/abs/2609.19927>
- **全文：** <https://arxiv.org/html/2609.19927v1>
- **项目页：** <https://wanglongzju.github.io/DirtyMoCap-Project-Page/>（[核查归档](../sites/dirtymocap.md)）
- **官方仓库：** <https://github.com/WangLongZJU/DirtyMoCap>（[核查归档](../repos/dirtymocap.md)）
- **一句话说明：** 将稀疏、含噪且无序的光学动捕点映射为固定身体代理锚点，再用可微 Gauss–Newton 拟合 SMPL-H，跨 marker layout 重建时序人体动作。

## 核心论文摘录（MVP）

### 1) 代理锚点：消除对 marker 标签与固定布局的依赖

- **链接：** 论文摘要、§1、§3.1。
- **摘录要点：** 现实光学 MoCap 会遇到 marker 遮挡丢失、重现后顺序变化、噪点/ghost marker，以及不同采集任务使用不同布点。DirtyMoCap 把原始无序 marker 转成身份固定的代理锚点（人体关节 + 身体表面点），建立稳定的 marker-to-body 中间表示；默认由 52 个 SMPL-H 关节和 61 个表面锚点组成 113 个锚点。
- **对 wiki 的映射：** [方法与局限](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)；[动作重定向流水线](../../wiki/concepts/motion-retargeting-pipeline.md)。

### 2) 滑窗跟踪 + 可学习的可微拟合

- **链接：** 论文 §3.2–§3.5。
- **摘录要点：** 首帧初始化锚点后，递归 16 帧窗口进行轨迹预测，窗口重叠一半并跨窗口保留 768 维隐藏状态。随后用可微 Gauss–Newton 求解器把轨迹拟合为 SMPL-H 姿态、全局平移与体型；输入相关的观测置信度、平滑权重和姿态先验权重由训练学习，而非固定人工调参。自定义 CUDA 雅可比实现相对 PyTorch 求解器报告最高 100× 加速。
- **对 wiki 的映射：** [核心原理与工程速查](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)。

### 3) 评测：跨布局合一模型，但 OOD 与实时性须谨慎

- **链接：** 论文 §4–§6。
- **摘录要点：** CMU/GRAB 合成评测包含 2,068 段序列，训练/测试按 80/20 划分；marker 布局覆盖 38–175 点，并注入缺失、重排、抖动、漂移、离群点等扰动。作者报告单模型在多种配置上达到约 8–10 mm MPJPE；但 SSM 真数据集上的对称 Chamfer 距离为 15.3 mm，较拥有 marker 专属先验的 MoSh++（11.3 mm）更差，而有效 marker-to-mesh 距离更低（7.5 vs 10.0 mm）。分阶段运行时间合计约 38.33 ms/frame，CUDA 加速声明不等于整条管线实时。
- **对 wiki 的映射：** [评测读法与边界](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)。

### 4) 武术动捕应用：标准化不等于数据已开放

- **链接：** 论文 §5 与官方项目页 “HKMALA-Motion Dataset”。
- **摘录要点：** 作者处理了 134 段传统武术光学 MoCap 序列，约每段 8,500 帧、累计约 180 分钟，含 22 个归一化武术风格标签，生成时序一致的 SMPL-H 动作。项目页截至核查日仍显示 “Data Coming soon”，因此这项数据应用不能解读为当前可下载数据集。
- **对 wiki 的映射：** [数据应用及发布状态](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)。

## 资料映射

- 论文及其方法/实验：[/wiki/entities/paper-dirtymocap-robust-motion-capture.md](../../wiki/entities/paper-dirtymocap-robust-motion-capture.md)
- 官方项目页和仓库开放状态：[/sources/sites/dirtymocap.md](../sites/dirtymocap.md)、[/sources/repos/dirtymocap.md](../repos/dirtymocap.md)
