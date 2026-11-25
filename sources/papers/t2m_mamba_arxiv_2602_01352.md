# T2M Mamba: Motion Periodicity-Saliency Coupling Approach for Stable Text-Driven Motion Generation

> 来源归档（ingest）

- **类型：** paper / text-to-motion / diffusion / state-space model
- **作者：** Xingzu Zhan、Chen Xie、Honghang Chen、Yixun Lin、Xiaochun Mai
- **机构：** 深圳大学、暨南大学
- **会议：** ICRA 2026（官方仓库及出版 DOI）
- **arXiv：** <https://arxiv.org/abs/2602.01352>
- **全文版本：** [arXiv v2 HTML](https://arxiv.org/html/2602.01352v2)；[PDF](https://arxiv.org/pdf/2602.01352v2)
- **DOI：** <https://doi.org/10.1109/ICRA57385.2026.11696326>
- **代码：** <https://github.com/chorl0229/T2Mmamba>；[源码归档](../repos/t2m-mamba.md)
- **入库日期：** 2026-11-25
- **通读：** 已完成
- **核对范围：** v2 HTML 摘要、§I–V（含式 1–13、表 I–IV、图注与结论）；官方仓 README、架构与源码审计文档、推理入口及核心模块。未运行 CUDA 复现实验。

## 核心摘录（归纳）

### 1. 周期与关键帧耦合（§III-B）

分段密度峰值检测得到关键帧及权重；用相邻关键帧划分时间区间，通过 FFT 自相关估计周期，并以峰值、显著度和谱熵过滤非周期段。非周期段以段长构造相位。相位与权重共同进入去噪网络，而非独立后处理。

对 wiki 的映射：[T2M Mamba](../../wiki/entities/paper-t2m-mamba.md)。

### 2. 状态空间与跨模态机制（§III-C/D）

相位编码加入运动输入，显著性调制 SSM 输入矩阵；PDCAM 对运动查询做相位旋转并以双路差分注意力减少无关响应。文本条件来自 CLIP，网络预测干净运动。

对 wiki 的映射：[T2M Mamba](../../wiki/entities/paper-t2m-mamba.md)。

### 3. 评测与边界（§IV/V）

HumanML3D FID 0.068、R-Top3 0.793；KIT-ML FID 0.195。去除两条路径中的关键帧与相位信息，HumanML3D FID 变为 0.139（表 IV）。改写鲁棒性主要是定性图示；实时应用属于未来工作。指标不是人形机器人真机跟踪成功率。

对 wiki 的映射：[T2M Mamba](../../wiki/entities/paper-t2m-mamba.md)。

## 对 wiki 的映射

- [独立论文实体](../../wiki/entities/paper-t2m-mamba.md)
