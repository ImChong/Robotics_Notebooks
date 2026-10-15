# AI-Predicted Full-Spectrum UV Correction Factors

- **类型：** paper
- **期刊：** *JACS Au* (2026)
- **DOI：** <https://doi.org/10.1021/jacsau.6c00812>
- **正式论文页：** <https://pubs.acs.org/doi/10.1021/jacsau.6c00812>
- **官方代码与数据：** <https://github.com/xtalpi-inc/lcms2yield>
- **官方介绍：** <https://www.xtalpi.com/ai-predicted-full-spectrum-uv-correction-factors-an-analyte-standard-free-framework-for-reaction-yield-quantification/>
- **入库日期：** 2026-10-15
- **一句话说明：** 将全光谱 UV 校正因子预测与 LC-UV 检测结合，支持在没有目标物纯标准品时估算化合物浓度和反应产率。

## 摘要摘录

高通量实验（HTE）常在早期缺少目标分析物的纯标准品，传统定量需要先制备、纯化和标定标准品，增加周期与成本。论文提出 “detect-predict” 流程：预测全光谱紫外校正因子（CF），再与实验 LC-UV 吸收信号结合，估算目标物浓度和反应产率。

模型采用两阶段训练：先用量子化学计算得到的 22,609 个化合物 CF 预训练以覆盖化学空间，再用 1,845 个实验 LC-UV 化合物样本微调，以适配真实仪器条件。论文报告，144 种 FDA 药物的浓度独立测试，以及 178 个真实反应（涵盖四类反应）的定量测试；后者相对校准曲线产率的 MAE 为 4.74%。另以 34 个 Suzuki–Miyaura 和 26 个 Minisci 反应对照分离产率，观察到与粗反应定量和纯化后收率差别相符的正偏差。

## 评测与解读

- 4.74% MAE 来自 178 个真实反应的产率定量对比，不应外推为所有反应类型和仪器上的通用误差保证。
- 分离产率会包含纯化过程损失；“粗反应液中的分析产率”与“纯化后分离产率”不是同一测量目标。
- 研究价值在于减少目标物标准品制备依赖，不能替代反应物/产物身份确认、色谱分离质量控制或方法学验证。

## 对 wiki 的映射

- 主实体：[AI 预测 UV 校正因子与无标准品反应定量](../../wiki/entities/paper-lcms2yield.md)
- 相关主题：[AI 设计物理实验](../../wiki/entities/paper-designing-physics-experiments-with-ai.md)

## 配套资料

- [XtalPi 官方介绍与开源状态核查](../sites/lcms2yield.md)
- [lcms2yield 代码与数据仓库](../repos/lcms2yield.md)
