# MiTA 官方代码

> 来源归档

- **仓库：** <https://github.com/QishuaiWen/MiTA>
- **许可证：** MIT
- **论文：** <https://arxiv.org/abs/2602.01219v6>
- **开放内容：** PyTorch 注意力模块，MiTA-DeiT、MiTA-ViT-5、MiTA-Segmenter 代码；README 列出检查点链接。
- **运行入口：** README 的 `from mita import MiTA_Attention`，依赖 flash-attn 2.6.3。示例最后调用的 `block(x)` 与前面的变量名不一致，使用前需核对。
- **核心实现：** `mita/mita_attention.py` 中 `MiTA_Attention.forward`、`mita`、`MixedAttention.apply`。
- **核查范围：** 已阅读 README 与核心注意力文件；尚未逐项核查训练入口及混合注意力底层实现。
- **沉淀到 wiki：** [MiTA](../../wiki/entities/paper-mita.md)
