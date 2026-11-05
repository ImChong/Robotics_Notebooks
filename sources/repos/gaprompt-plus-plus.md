# GAPrompt++ 官方实现

- **代码：** https://github.com/PKU-OV3-LAB/GAPromptPlus
- **论文归档：** [GAPrompt++](../papers/gaprompt_plus_plus_2609_19716.md)
- **数据：** [GSModel60](https://huggingface.co/datasets/zxAi/GSModel60)；[uCO3D80](https://huggingface.co/datasets/zxAi/uCO3D80)
- **核查日期：** 2026-11-04

## 官方 README 核查

仓库提供对象级分类、部件分割与场景级语义分割代码和配置；README 分别给出 `object_level/main.py` 与 `scene_level/main.py` 的运行示例。官方说明部分上游权重可下载，Point-PQAE、Utonia 和 GAPrompt++ 下游 checkpoint 标记为待发布；使用前应重新核对仓库状态。场景级代码保留其独立许可证，仓库级许可证尚待作者批准。

官方 README 称 GSModel60 含 60 类、13,847 个点云，uCO3D80 含 80 类、20,464 个点云。数据版本与切分细节有待对照论文和数据卡核实。

## 待完成

尚未审读入口脚本与相关模块的实际调用链；源码运行时序图待核实后补充。尚未执行代码或验证复现结果。
