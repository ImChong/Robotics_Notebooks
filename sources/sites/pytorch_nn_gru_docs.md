# PyTorch `torch.nn.GRU` 官方文档

> 来源归档（ingest）

- **标题：** torch.nn.GRU — PyTorch Stable Documentation
- **类型：** site / official-api-docs
- **URL：** <https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html>（稳定版重定向至 2.x 文档树）
- **组织：** PyTorch / Linux Foundation
- **入库日期：** 2026-09-21
- **一句话说明：** PyTorch 多层 **GRU** 模块的官方 API、门控更新公式、张量形状约定，以及 **与 Cho 原论文 reset 乘法顺序差异** 的实现说明。

## API 要点（截至 2026-09-21 文档）

| 参数 | 含义 |
|------|------|
| `input_size` | 输入特征维 $H_{in}$ |
| `hidden_size` | 隐状态维 $H_{out}$ |
| `num_layers` | 堆叠层数（下层输出作上层输入，层间 dropout） |
| `batch_first` | `True` 时张量为 `(N,L,H)` 而非默认 `(L,N,H)` |
| `bidirectional` | 双向 GRU，输出维 $D=2$ |

**单步更新（文档公式）：**

$$
\begin{aligned}
r_t &= \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{t-1} + b_{hr}) \\
z_t &= \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{t-1} + b_{hz}) \\
n_t &= \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1}+b_{hn})) \\
h_t &= (1-z_t)\odot n_t + z_t \odot h_{t-1}
\end{aligned}
$$

**实现差异（文档 Note）：** 原论文与其它框架常在 **$W_{hn}$ 之前** 做 $r_t \odot h_{t-1}$；PyTorch 为效率在 **$W_{hn}h_{t-1}+b_{hn}$ 之后** 再 Hadamard，复现文献数值时需知晓此点。

**最小示例：**

```python
rnn = nn.GRU(10, 20, 2)
input = torch.randn(5, 3, 10)   # (L, N, H_in)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
```

## 开源状态

**已开源** — PyTorch 核心库 Apache-2.0；文档随版本发布。

## 对 wiki 的映射

- [`wiki/concepts/gru.md`](../../wiki/concepts/gru.md) — 工程实现与公式对照
- [`wiki/entities/pytorch.md`](../../wiki/entities/pytorch.md) — 框架实体
- [`sources/repos/pytorch-official.md`](../repos/pytorch-official.md) — 站点总索引
