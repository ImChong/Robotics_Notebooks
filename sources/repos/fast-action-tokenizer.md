# FAST / FAST+ 与 π₀-FAST — 官方源码核对

- **类型：** 官方源码 / tokenizer 资产 / 自回归 VLA
- **论文：** https://arxiv.org/abs/2501.09747
- **项目页（非源码）：** https://www.pi.website/research/fast
- **FAST+ 源码及资产：** https://huggingface.co/physical-intelligence/fast/tree/main
- **核心文件：** https://huggingface.co/physical-intelligence/fast/blob/main/processing_action_tokenizer.py
- **π₀-FAST JAX 源码：** https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_fast.py
- **π₀-FAST 接线代码：** https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py
- **核查日期：** 2026-11-08
- **核对范围：** 完整阅读 HF `README.md` 与 `processing_action_tokenizer.py`；查阅 `openpi` 的 `pi0_fast.py`、`tokenizer.py`、`policies/policy.py`、`policies/policy_config.py`、`training/config.py` 中与 π₀-FAST 训练和推理有关的函数；查阅官方 openpi README 模型表。没有运行模型、下载 checkpoint 或真机测试。
- **通读：** 已完成

## 资产与开源边界

| 组件 | 官方来源 | 核实结果 |
| --- | --- | --- |
| FAST+ 通用动作分词器 | `physical-intelligence/fast` | `UniversalActionProcessor`、`tokenizer.json`、`processor_config.json`、自定义训练 `.fit()` 公开 |
| π₀-FAST 自回归策略 | `Physical-Intelligence/openpi` | `Pi0FAST`、`FASTTokenizer`、训练配置、推理路径与公开 checkpoint 入口 |
| 原论文完整训练数据 | 官方研究页及开源 README | 未发现完整原始数据公开清单；不能由有 checkpoint 推断数据全开放 |

## 核心函数（直接对照源码）

1. `UniversalActionProcessor.__call__(action_chunk)`：二维输入扩为 batch，缓存 `time_horizon`、`action_dim`；`scipy.fft.dct(..., axis=1, norm="ortho")`，乘 `scale` 并四舍五入，展平系数经 `bpe_tokenizer(...)["input_ids"]` 生成 token。
2. `UniversalActionProcessor.decode(tokens, time_horizon, action_dim)`：BPE 解码、恢复整数 DCT 系数矩阵，再通过 `idct(..., axis=0, norm="ortho")` 恢复动作。**如 token 解码或 shape 不符合预期，源码打印错误并退回全零 DCT 系数**，应主动检测错误以免将零动作误作正常动作。
3. `UniversalActionProcessor.fit(action_data)`：沿时间轴 DCT、找系数整数取值范围、用 `BpeTrainer` 训练新的 BPE 词表，返回新 processor；输入需预先按相同动作语义和归一化口径处理。
4. `openpi.models.tokenizer.FASTTokenizer.tokenize`：状态分为 256 个 bin，以 `Task/State/Action` 提示组织，FAST 动作 tokens 映射到 PaliGemma 词表末尾，保留特殊 token 区。
5. `openpi.models.pi0_fast.Pi0FAST.compute_loss`：基于 prefix-LM 掩码对动作 token 做 next-token 交叉熵；`sample_actions`：prefix KV-cache 预填充后自回归循环直到 EOS 或 `max_decoding_steps`。
6. `openpi.policies.policy.Policy.infer`：输入 transforms → `model.sample_actions` → 输出 transforms，后者通过 `ExtractFASTActions` 调用 `FASTTokenizer.extract_actions` 转回连续动作，附推理时间字段 `policy_timing.infer_ms`。

## 工程验证注意

- 官方文档用 `AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)`，而**不是**将 tokenizer 当成 `AutoModel` 直接运行；仅信任经过核验的远程 Python 源码。
- `FASTTokenizer.extract_actions` 若生成文本缺少 `Action: `，源码直接返回全零动作；部署时监控解析失败、token 数、动作重建误差与真实推理时延。
- 论文报告的单块推理（RTX 4090）约为 π₀-FAST 750ms 对 π₀ flow 100ms；属于论文测试条件，不代表任意硬件实测。

## 对 wiki 的映射

- [FAST / π₀-FAST 论文实体](../../wiki/entities/paper-rcl-2501-09747-fast-efficient-action-tokenization-for-vision-la.md)
- [项目官网来源](../sites/fast-action-tokenization.md)
- [论文全文归档](../papers/fast_action_tokenization_arxiv_2501_09747.md)
- [openpi 现有仓库归档](openpi.md)
