# Fiatlux 项目页

- **URL：** <https://fiatlux-bench.github.io/>
- **论文：** <https://arxiv.org/abs/2609.38216>；[归档](../papers/fiatlux_arxiv_2609_38216.md)
- **代码：** <https://github.com/haw-ai-i/fiatlux>；[归档](../repos/fiatlux.md)
- **遥操作数据：** <https://huggingface.co/datasets/haw-ai-i/fiatlux-teleoperation>；[归档](../datasets/fiatlux-teleoperation.md)
- **资产：** <https://huggingface.co/datasets/haw-ai-i/fiatlux-assets>
- **基线记录：** <https://huggingface.co/datasets/haw-ai-i/fiatlux-policy-baselines>
- **核查日期：** 2026-10-02（实际打开项目页、README 与数据卡）
- **实体页：** [Fiatlux](../../wiki/entities/paper-fiatlux.md)

## 开源状态

| 组件 | 核查结果 |
|---|---|
| Code | 公开训练、运行、遥操作、记录与离线评分脚本；Apache-2.0，Isaac Lab 派生文件另受 BSD-3 许可 |
| Teleoperation | 仿真示范记录；CC-BY-4.0；不是 policy checkpoint |
| Assets | USD 资产数据入口；第三方资产各有许可，不统一视为 Apache-2.0 |
| Policy baselines | zero/random/GR00T 的 rollout 数据；不是已训练的换灯策略权重 |
| 攀爬示范 | 四个攀爬子任务未展示通过门控的 takes；站在梯上不代表完成上下梯 |
| 真机部署 | G1 SDK 适配器仍待实现 |

结论：**基准代码与数据已公开，完整攀梯换灯能力未解决。** 项目页直接标注八项有成功示范、四项仍欠验证，归档保留这个边界。
