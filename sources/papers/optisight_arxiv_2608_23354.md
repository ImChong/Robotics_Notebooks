# OptiSight（arXiv:2608.23354 / IRI-TR-26-02）

> 来源归档（ingest）— 根据 arXiv 摘要、IRI 官方出版记录和作者仓库 README 整理，不是原文镜像。

- **论文标题：** OptiSight: Bridging Semantic Reasoning and Geometric Control for Embodied Navigation
- **作者：** Alperen Avan、Jordi Sanchez-Riera
- **机构：** Institut de Robòtica i Informàtica Industrial（IRI，CSIC-UPC），西班牙巴塞罗那
- **类型：** 具身室内导航 / 视觉语言推理 / 视觉伺服
- **论文：** [arXiv:2608.23354](https://arxiv.org/abs/2608.23354)
- **机构出版记录：** [IRI-TR-26-02](https://www.iri.upc.edu/publications/show/3160)
- **代码：** [OptiSight-Python-Multimodal-CoT-for-Visual-Reasoning](https://github.com/avanalperen/OptiSight-Python-Multimodal-CoT-for-Visual-Reasoning)
- **一句话说明：** OptiSight 用 VLM 在关键时刻处理目标语义，再以 Grounded-SAM 与相机投影几何形成连续导航控制，避免让大模型持续承担低层运动控制。

## 论文要点

1. **语义与几何分工。** VLM 负责解释目标和关键决策，视觉定位与几何控制负责把图像中的目标转成可执行的导航命令。
2. **按需调用推理。** 有限状态 Chain-of-Thought 架构只在关键决策点请求 VLM；连续移动由确定性视觉伺服处理，以压低持续推理开销。
3. **开放词汇定位。** Grounded-SAM 用于定位开放词汇目标；相机投影几何将视觉观测映射为导航控制，不要求稠密建图。
4. **仿真验证。** 论文在 AI Habitat 中报告零样本室内导航，覆盖障碍规避与语义歧义场景，并称可在 8 GB 显存预算内运行。

## 复现提示

作者公开了 Python 仓库，可从 `start.py` 启动本地 dashboard。README 说明运行前需要单独准备 AI Habitat 地图及分割、检测和 VLM 模型文件；这些大型资源并未随仓库提交。仓库也明确说明没有自动计算汇总成功率的评测脚本：24 个实验场景的每场 10 次重复测试由人工记录和统计。因此应把论文汇总指标与仓库中可自动复跑的评测能力分开看。

## 对 wiki 的映射

- [OptiSight 论文实体页](../../wiki/entities/paper-optisight-embodied-navigation.md)
- [视觉–语言导航任务页](../../wiki/tasks/vision-language-navigation.md)
- [视觉伺服方法页](../../wiki/methods/visual-servoing.md)
- [IRI 官方出版记录](../sites/optisight-iri-tr-26-02.md)
- [作者代码仓库](../repos/optisight-python-multimodal-cot.md)
