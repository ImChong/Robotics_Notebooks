# cowarobot-ai.github.io/RISE（项目页）

> 来源归档（ingest）

- **标题：** RISE: Adaptive Imagination for World Action Models
- **类型：** site / project-page
- **官方入口：** <https://cowarobot-ai.github.io/RISE/>
- **入库日期：** 2026-09-02
- **一句话说明：** 酷哇科技 WAM 系列第二项（前作 DAWN）：自适应 Roll/Stop 想象调度 + CounterDrive；页内给出论文数值、代码与数据集链接。

## 页面公开信息（检索自 2026-09-02）

| 资源 | URL / 状态 |
|------|------------|
| 项目首页 | <https://cowarobot-ai.github.io/RISE/> |
| arXiv | <https://arxiv.org/abs/2608.20430>（BibTeX `eprint={2608.20430}`） |
| **代码** | **已开源** — <https://github.com/COOWAI/RISE> |
| **数据集** | **已发布** — <https://huggingface.co/datasets/COWARobot/CounterDrive> |
| 作者单位 | COWARobot Co. Ltd / Shanghai Jiao Tong University / Hohai University |
| 通讯 | volans.liao@cowarobot.com、pengpai@cowarobot.com |
| 系列 | 「RISE is the second project in COWARobot's World Action Model research series. See the preceding work, DAWN.」 |

## 页内主张（便于 wiki 溯源）

1. 固定想象预算 → 按场景的 Roll/Stop：Latent Evaluator（风险 + Future Planning Gain）与 Rollout Gate（增益 vs 算力）。
2. CounterDrive：NAVSIM 5013/1000、nuScenes 2432/511；人工核验 ego 一致性、事故起始、因果类。
3. 数字：NAVSIM v1 PDMS **91.5**、v2 EPDMS **90.8**、nuScenes Avg. L2 **0.31 m**、平均 rollout **2.40**、延迟 **287.429 ms**。
4. 消融：Scheduler 与 CounterDrive 合用最好；Random Stop / Latent Margin 都打不过 Scheduler。

## 对 wiki 的映射

- [`wiki/entities/paper-rise-adaptive-imagination-wam.md`](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)
- [`sources/papers/rise_adaptive_imagination_arxiv_2608_20430.md`](../papers/rise_adaptive_imagination_arxiv_2608_20430.md)
- [`sources/repos/coowai-rise.md`](../repos/coowai-rise.md)
- [`sources/datasets/counterdrive.md`](../datasets/counterdrive.md)
