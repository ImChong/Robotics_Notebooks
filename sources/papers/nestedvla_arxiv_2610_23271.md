# NestedVLA（arXiv:2610.23271）

> 来源归档（paper）

- **标题：** NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.23271>
- **PDF：** <https://arxiv.org/pdf/2610.23271>
- **代码：** <https://github.com/WanyuGroup/NeurIPS2026-NestedVLA>
- **入库日期：** 2026-11-01
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-01）：仓库可达（git ls-remote）。

## 核心摘录（公众号转述，未核验原文表格）

- 不直接合并权重，而是按任务生成技能。
- NestedVLA用超网络生成动作专家参数，并单独保存压缩的视觉语言残差；三个仿真基准支持减少技能冲突与存储，尚不能视为任意新技能的零样本组合。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-nestedvla](../../wiki/entities/paper-nestedvla.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)
