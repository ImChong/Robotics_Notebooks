# FAST：Efficient Robot Action Tokenization（官方研究页）

- **类型：** 官方研究介绍 / 项目主页
- **链接：** https://www.pi.website/research/fast
- **发布：** 2025-01-16（Physical Intelligence）
- **论文：** https://arxiv.org/abs/2501.09747
- **官方 FAST+ tokenizer 代码与资产：** https://huggingface.co/physical-intelligence/fast
- **官方 π₀-FAST 模型代码：** https://github.com/Physical-Intelligence/openpi
- **核查日期：** 2026-11-08
- **核对范围：** 已阅读官网研究页（动机、DCT+BPE 图示、训练/控制对照、FAST+ 调用和局限），并对照论文全文与上述官方源码入口。
- **通读：** 已完成

## 核实结论

1. **官网不是代码仓库。** 用户给出的 `pi.website/research/fast` 是作者发布的研究介绍页；真正可读的 FAST+ 实现及 tokenizer 资产位于 Hugging Face 的 `physical-intelligence/fast`。
2. **方法：** 动作块先做 DCT、系数量化、低频优先展平，再用 BPE 将重复符号压缩为自回归动作 token；官网称一般每块约 30–60 token，与逐维逐步分箱相比可大幅缩短序列。
3. **开源：** FAST+ 的 `processing_action_tokenizer.py`、BPE tokenizer 配置/资产公开；官方 `openpi` 提供 π₀-FAST VLA 的模型、训练/推理配置及 checkpoint。**“tokenizer 已开放”与“论文全部原始训练数据及实验可完整复现”不是一回事。**
4. **性能边界：** 官网和论文都报告训练速度最高提升约 5 倍，但官网也明确指出，π₀-FAST 的逐 token **推理**速度明显慢于 flow-matching π₀，不能把训练收益当成在线控制速度收益。

## 对 wiki 的映射

- [FAST / π₀-FAST 主节点](../../wiki/entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md)
- [论文全文归档](../papers/fast_action_tokenization_arxiv_2501_09747.md)
- [官方代码归档](../repos/fast-action-tokenizer.md)
