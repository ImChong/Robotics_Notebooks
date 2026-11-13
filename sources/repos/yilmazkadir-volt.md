# Volt 官方实现（YilmazKadir/Volt）

- **URL：** <https://github.com/YilmazKadir/Volt>
- **项目页：** <https://yilmazkadir.github.io/Volt/>
- **论文：** [arXiv:2604.19609v2](../papers/volt_arxiv_2604_19609.md)
- **权重：** <https://huggingface.co/KadirYilmaz/Volt>
- **许可证：** MIT（官方仓库）
- **入库/复核日期：** 2026-11-13
- **状态：** 已开源可运行实现（实际运行仍依赖数据集、CUDA、Python 环境）。

## 源码入口与运行顺序（已核查官方文件）

| 环节 | 实际位置 | 作用 |
|---|---|---|
| CLI | [scripts/train.sh](https://github.com/YilmazKadir/Volt/blob/main/scripts/train.sh) | 解析 -d/-c/-n/-g；写入 exp 并执行训练脚本 |
| Launcher | [tools/train.py](https://github.com/YilmazKadir/Volt/blob/main/tools/train.py) | default_config_parser → launch → main_worker |
| Trainer | [pointcept/engines/train.py](https://github.com/YilmazKadir/Volt/blob/main/pointcept/engines/train.py) | build model / dataloader / optimizer；run_step 前向与反向 |
| Segmentor | [pointcept/models/default.py](https://github.com/YilmazKadir/Volt/blob/main/pointcept/models/default.py) | DefaultSegmentorV2: backbone + 分类头/可选 teacher |
| Backbone | [pointcept/models/volt/volt_base.py](https://github.com/YilmazKadir/Volt/blob/main/pointcept/models/volt/volt_base.py) | Tokenizer → 3D RoPE → Transformer blocks → Decoder |
| Decoder | [pointcept/models/volt/decoder.py](https://github.com/YilmazKadir/Volt/blob/main/pointcept/models/volt/decoder.py) | 将非空 patch 的 token 映回细粒度点特征 |

## 官方 README 训练示例

```bash
# 需按 README 配置 CUDA 12.6、uv 与数据集；蒸馏实验还需 teacher weights
uv sync
hf download KadirYilmaz/Volt --include "teacher_weights/*.pth" --local-dir weights/
sh scripts/train.sh -g 4 -d scannet -c semseg-volt-distill -n semseg-volt-distill
```

**说明：** 该命令是监督分割 + 卷积教师蒸馏训练 Volt，而非 VINO 的自监督训练。VINO 尚无官方可运行入口。

## 交叉链接

- [Volt wiki 节点](../../wiki/entities/paper-volt.md)
- [Volt 项目页](../sites/volt-project.md)
- [VINO 方法归档](../papers/vino_arxiv_2611_07176.md)
