# TexasPokerRobot（Hugging Face）

> 来源归档

- **标题：** TexasPokerRobot — DexHoldem 真机扑克桌面示范
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot>
- **论文：** [arXiv:2605.18727](https://arxiv.org/abs/2605.18727)
- **项目页：** <https://dexholdem.github.io/Dexholdem/>
- **机构：** 香港大学（HKU）
- **入库日期：** 2026-09-05
- **访问条件：** 公开；原始 `.npz` 约 378 GB，Viewer 只显示 manifest
- **一句话说明：** 14 个扑克原语 × 105 条遥操作 episode；三路 RGB-D + 关节状态；CC BY 4.0。

---

## 规模与结构

| 原语 | episode | 原始体积 |
|------|---------|----------|
| `pick_up_left` / `pick_up_right` | 105 / 105 | 40.19 / 29.63 GB |
| `push_5` / `push_10` / `push_50` / `push_100` | 各 105 | 28.44–35.04 GB |
| `pull_5` / `pull_10` / `pull_50` / `pull_100` | 各 105 | 29.00–31.83 GB |
| `put_down_left` / `put_down_right` | 105 / 105 | 18.23 / 14.59 GB |
| `show_left` / `show_right` | 105 / 105 | 13.23 / 12.19 GB |
| **合计** | **1,470** | **约 378 GB** |

论文训练划分：每原语 **100 训 / 5 验**，由 [Dexholdem-Policy](https://github.com/DexHoldem/Dexholdem-Policy) 的 `workflow/organize_data.py --eval_count 5` 落地。HF 只提供 `data/train.csv` manifest，不另给官方 test split。

加载：

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np

ds = load_dataset("Winniechen2002/TexasPokerRobot")
row = ds["train"][0]
path = hf_hub_download(
    repo_id="Winniechen2002/TexasPokerRobot",
    filename=row["file_path"],
    repo_type="dataset",
)
episode = np.load(path, allow_pickle=True)
```

`allow_pickle=True` 是因为部分关节记录存成 object array；只对信任的官方文件使用。

## 局限

- 体积大，不适合当「随手拉的小集」。
- Viewer 看不到 RGB/深度张量。
- 失败尝试已按原语成功条件剔除，发布集是成功示范。

## 关联资料

- 论文：[`sources/papers/dexholdem_arxiv_2605_18727.md`](../papers/dexholdem_arxiv_2605_18727.md)
- 项目页：[`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- 策略仓：[`sources/repos/dexholdem-policy.md`](../repos/dexholdem-policy.md)
- Wiki：[wiki/entities/paper-dexholdem.md](../../wiki/entities/paper-dexholdem.md)
