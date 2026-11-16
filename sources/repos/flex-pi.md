# Flex-π（geyan21/flex-pi）

> 来源归档（repo）

- **标题：** Flex-π — A Multi-Stream World-Action Model with Compute Flexibility
- **类型：** repo / world-action-models / joint-wam / placeholder
- **来源：** geyan21（UW）
- **链接：** <https://github.com/geyan21/flex-pi>
- **论文：** [arXiv:2608.10860](https://arxiv.org/abs/2608.10860) — [`sources/papers/flex_pi_arxiv_2608_10860.md`](../papers/flex_pi_arxiv_2608_10860.md)
- **项目页：** <https://flex-pi.github.io/> — [`sources/sites/flex-pi-github-io.md`](../sites/flex-pi-github-io.md)
- **入库日期：** 2026-08-13
- **一句话说明：** 官方代码占位仓；README 声明训练/推理与 checkpoint「ready soon」。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-flex-pi.md`](../../wiki/entities/paper-flex-pi.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-13 复核） |
|----|-------------------|
| 训练 / 推理代码 | **待发布**（GitHub API：`size=1`，contents 仅 `README.md` ~2.2 KB） |
| 权重 | 未发布 |
| 许可证 | 未声明 SPDX |
| last push | 2026-08-12T08:16:49Z |
| README 承诺 | 「The code is ready soon… Watch the repo」 |

**结论：** **宣称将开源 / 代码待发布**；勿写「已开源可运行」。独立 ingest 复核与首轮批量入库结论一致。

## 关联资料

- 论文：[`sources/papers/flex_pi_arxiv_2608_10860.md`](../papers/flex_pi_arxiv_2608_10860.md)
- 项目页：[`sources/sites/flex-pi-github-io.md`](../sites/flex-pi-github-io.md)
- Wiki：[wiki/entities/paper-flex-pi.md](../../wiki/entities/paper-flex-pi.md)

## 固定版本源码核读（2026-11-13）

- **版本：** [84d9c5bfdbf4b771938c1bf7f3a79909620ccae1](https://github.com/geyan21/flex-pi/tree/84d9c5bfdbf4b771938c1bf7f3a79909620ccae1)
- **核对范围：** 根README全文；serve_yam_flexpi.main、server_adapter.YamFlexPiServerPolicy.infer、deploy_policy.YamFlexPiPolicy.infer_action_chunk、FlexPi._infer_action_dispatch完整函数。
- **通读：** 已完成
- 训练/评测/部署与多套权重和YAM数据已发布；10月preview约4600h/760h与论文500h训练不同版本，34D混合本体/LoRA扩展不混入论文32D协议。
- WebSocket adapter将flat观测转nested，合并runtime masks；录制预测会强制全部joint，改变时延。YAM策略拼接RGB、按present_pointmap决定是否必须深度与K，缓存文本并归一化state→model.infer_action→反归一化/相对到绝对动作。可选speed/smoother后处理与模型原始输出分开返回。
- 无joint且未force_joint时走KV缓存动作路径，否则joint路径；action-only不等于输入只RGB。服务端current预热断言32D，不能直接套10月34D预览而不按对应文档核对。
- MIT代码/发布权重与底层DINO/训练数据条款分开。静态阅读，未加载GPU或真机执行。
