# kyutai-labs/reki

- **类型：** 官方代码仓库
- **仓库：** <https://github.com/kyutai-labs/reki>
- **项目页：** <https://kyutai.org/reki/>
- **模型集合：** <https://huggingface.co/collections/kyutai/reki-6ac371366b3da8cdece73ce2>
- **论文：** [arXiv:2610.22178](https://arxiv.org/abs/2610.22178)
- **代码许可：** MIT
- **权重许可：** CC-BY-NC-SA 4.0
- **源码状态：** 数据整理、训练、推理、评测、部署和权重公开（核验日期：2026-10-19）
- **知识节点：** [REKI](../../wiki/entities/paper-reki-vla-register-memory.md)

## README 入口

- 模型：Qwen3.5-2B VLM + DiT action expert + flow matching；32 个 register tokens 经 expert KV cache 保存短期信息。
- 评测：仓库附有训练/评估脚本与基准配置；编译后 H100 单动作块推理时间约 21.8 ms（README 报告）。
- 注意许可：模型权重的非商业许可不同于 MIT 代码许可。

## 关联资料

- [项目页归档](../sites/reki.md)
- [论文摘录](../papers/reki_arxiv_2610_22178.md)
