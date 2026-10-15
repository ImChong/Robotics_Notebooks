# lcms2yield 官方介绍页

> 项目资料归档

- **项目：** lcms2yield — AI-Predicted Full-Spectrum UV Correction Factors
- **类型：** site / 公司研究介绍
- **网址：** <https://www.xtalpi.com/ai-predicted-full-spectrum-uv-correction-factors-an-analyte-standard-free-framework-for-reaction-yield-quantification/>
- **论文：** [JACS Au 2026, DOI 10.1021/jacsau.6c00812](https://doi.org/10.1021/jacsau.6c00812)
- **代码与数据：** <https://github.com/xtalpi-inc/lcms2yield>
- **入库日期：** 2026-10-15
- **一句话说明：** XtalPi 对无分析物标准品反应产率定量研究的官方介绍；资料页关联论文与公开实现。
- **开源核查：** 已开源。官方 GitHub 仓库公开推理代码和研究数据，仓库 README 标注 Apache-2.0；模型使用方法见 README。
- **归属实体：** [wiki/entities/paper-lcms2yield.md](../../wiki/entities/paper-lcms2yield.md)
- **资料映射：** [论文摘录](../papers/lcms2yield_jacsau_2026.md) · [仓库归档](../repos/lcms2yield.md)

## 研究定位

介绍页描述由全光谱 UV 校正因子预测支持的 “detect-predict” 定量流程：量化计算预训练提供分子覆盖，实验 LC-UV 数据微调以贴近仪器条件，预测因子与色谱 UV 信号配合用于估计目标物浓度及反应产率。该方向服务高通量合成与化学发现，可与自动化实验平台结合；它本身不是机器人控制算法。

## 核查备注

代码开放结论依据页面引用的官方仓库及其 README，而非仅依据论文中的开放声明。仓库 README 给出从 SMILES 列表预测 CF 的接口；公开页面未提供通用 LC 仪器采集驱动或完整自动化实验站集成。
