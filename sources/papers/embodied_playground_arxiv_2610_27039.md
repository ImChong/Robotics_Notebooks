# Embodied Playground | arXiv:2610.27039v2

- **类型：** paper
- **论文：** [摘要](https://arxiv.org/abs/2610.27039v2) · [PDF](https://arxiv.org/pdf/2610.27039v2) · [HTML](https://arxiv.org/html/2610.27039v2)
- **版本：** v1 2026-10-19；本次核对 v2 2026-10-25。
- **作者：** Feng Chen、Pei Zhou、Lixuan He、Yunchao Yao、Li Sun、Shenghua Gao、Mingyu Ding、Kaifeng Zhang、Kaichen Zhou、Yi Ma。
- **机构署名：** 香港大学、Sharpa（忆生科技）、清华大学、斯坦福大学、北卡罗来纳大学教堂山分校、MIT、Transcengram。
- **项目页：** <https://winniechen2002.github.io/Agent-bench-website/>
- **代码：** <https://github.com/Winniechen2002/Embodied-Playground>
- **核对范围：** v2 正文 §1–7，附录 A 的任务目录、难度规则、失败类型和四个子集索引。
- **通读：** 已完成
- **最初入库 / 本次核对：** 2026-10-31 / 2026-11-25
- **对应节点：** [Embodied Playground](../../wiki/entities/paper-embodied-playground.md)

1. §3–4：410 个任务与本体组合，EP-100 仅取其中 100 个；EP-20 是诊断集，不是全榜的无偏样本。
2. §5：公共 SDK 的读操作不推进物理；动作推进仿真；finish 只评价现有证据。独立评分包含轨迹中的规则违规、终态目标与适用的稳定性。
3. §6 与表 5：5 套件等权，GPT-6-Astra 13.80/19.94、Opus-5.5-extra 29.22/35.85（Pass@1/2，百分比）。原始条件汇总和每次有效尝试分母另列。
4. §6.3：审计分开保存原生评分与规则披露缺陷。GPT-6-Astra 未获得完整 token 用量，不能把允许上限当作实测。

[项目页归档](../sites/embodied-playground.md) · [源码归档](../repos/embodied_playground.md)
