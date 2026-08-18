# Behavior Prompting Policy: Demonstrations as Prompts for Manipulation

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** Behavior Prompting Policy: Demonstrations as Prompts for Manipulation
- **类型：** paper
- **状态：** arXiv 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.30457>
  - 项目页：https://behavior-prompting.github.io/
- **代码：** https://github.com/real-stanford/behavior_prompting
- **作者：** Austin Patel, Ben Pekarek, Joel Enrique Castro Hernandez, Shuran Song
- **机构：** Stanford University; UC Berkeley
- **入库日期：** 2026-08-18
- **一句话说明：** BPP（arXiv:2606.30457）：单次人类示范作 in-context behavior prompt；iPhUMI 采集多样数据；DrawAnything/LIBERO-Gen 评测；训练与部署代码已开源。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 把单次人类示范当作测试时 prompt，与当前观测一起输入 Transformer 扩散策略，无需微调即可执行新任务或定义新能力。
- **对 wiki 的映射：**
  - [wiki/entities/paper-behavior-prompting-policy.md](../../wiki/entities/paper-behavior-prompting-policy.md)

### 方法与结果（归纳）

- **方法：** prompt 编码器 + 扩散动作解码器；训练数据任务多样性是关键；iPhUMI 无线传输测试示范。
- **评测：** DrawAnything 绘制与 LIBERO-Gen 桌面操作；测试时单示范泛化未知任务。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-behavior-prompting-policy.md`
