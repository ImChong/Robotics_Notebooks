# MimicAll: Structured Embodiment Conditioning for Cross-Humanoid Whole-Body Motion Tracking

- **类型：** paper
- **论文：** <https://arxiv.org/abs/2610.33508>（2026-10-24 提交）
- **作者：** Jia Li、Lingyun Xu、Yitang Li、Hao Zhao
- **入库日期：** 2026-10-28
- **一句话说明：** 将机体结构描述编码为全局和逐关节条件，与观测历史及匹配机体的参考动作一起预测关节目标偏移。

## 摘录

- 问题与方法：将机体结构描述编码为全局和逐关节条件，与观测历史及匹配机体的参考动作一起预测关节目标偏移。
- 评测或数据：训练覆盖 16 类人形机器人的 4096 个变体；随机 G1 的 100STYLE 评测中，共享训练 30 秒成功率为 99.4%，单机体专家为 42.9%。
- 边界：数值限于论文所述数据集、随机 G1 和评测设置；不能直接推断真机效果。未核实到官方代码入口。

## 对 wiki 的映射

- [跨本体全身动作跟踪](../../wiki/entities/paper-mimicall.md)
