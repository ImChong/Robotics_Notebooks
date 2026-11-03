# GAPrompt++: Multi-Granular Geometry-Aware Point Cloud Prompt for 3D Vision Model

> 待完成的来源归档：目前仅核对论文摘要、官方代码 README 和数据集页面，尚未通读论文全文。

- **论文：** [IEEE Xplore](https://ieeexplore.ieee.org/document/11676080) · [arXiv 2609.19716](https://arxiv.org/abs/2609.19716) · [DOI](https://doi.org/10.1109/TPAMI.2026.3729984)
- **作者：** Zixiang Ai、Zhenyu Cui、Yufei Guo、Wenwen Qiang、Lei Chen、Jiwen Lu、Jiahuan Zhou
- **机构：** 北京大学、清华大学、中国航天科工集团智能科技研究院、中国科学院软件研究所（依官方仓库）
- **发表：** IEEE Transactions on Pattern Analysis and Machine Intelligence，2026
- **官方源码：** [PKU-OV3-LAB/GAPromptPlus](https://github.com/PKU-OV3-LAB/GAPromptPlus)；[仓库归档](../repos/gaprompt-plus-plus.md)
- **数据集：** [GSModel60](https://huggingface.co/datasets/zxAi/GSModel60)；[uCO3D80](https://huggingface.co/datasets/zxAi/uCO3D80)

## 待核验的摘要信息

官方摘要介绍了 Point Shift Prompter、Keypoint Prompter 和 Prompt Propagation，用于预训练点云模型的参数高效适配。官方仓库称评测覆盖对象分类、部件分割和场景语义分割。具体机制、实验结果与适用边界须对照全文核实。

## 未完成事项

- IEEE 与 arXiv 全文入口在本次环境中无法获取；需要通读论文及相关附录，核对方法、实验、局限和版本。
- 阅读训练或推理入口及其调用链，补充源码运行时序图。
- 完成全文核验后创建项目实体页，补 Mermaid 项目流程图、关联链接及结论，并运行仓库的 `make ci-preflight`。
