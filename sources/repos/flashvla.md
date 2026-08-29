# FlashVLA（z-lab/flashvla）

> 来源归档

- **标题：** FlashVLA
- **类型：** repo
- **来源：** z-lab（论文作者 UCSD / MIT）
- **链接：** <https://github.com/z-lab/flashvla>
- **论文：** <https://arxiv.org/abs/2608.27384>
- **博客：** <https://z-lab.ai/projects/flashvla/>
- **权重集合：** <https://huggingface.co/collections/z-lab/flashvla>
- **LIBERO 权重：** <https://huggingface.co/z-lab/flashvla-pi05-libero>
- **RoboTwin 权重：** <https://huggingface.co/z-lab/flashvla-pi05-robotwin>
- **许可：** Apache-2.0
- **入库日期：** 2026-08-29
- **一句话说明：** 流匹配 VLA 的流式动作解码实现：训练、LIBERO/RoboTwin 异步评测、延迟基准与真机部署；基于 LeRobot 与 VLASH。
- **沉淀到 wiki：** [`wiki/entities/paper-flashvla.md`](../../wiki/entities/paper-flashvla.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `conda env create -f environment.yml`；LIBERO / RoboTwin 另需 `sim_eval/` 仿真环境 |
| LIBERO 评测 | `bash sim_eval/libero/eval.sh`（权重 `z-lab/flashvla-pi05-libero`） |
| RoboTwin 2.0 | 服务端 `eval_server.sh`（flashvla env）+ 客户端 `eval_client.sh`（RoboTwin env） |
| 训练 | `bash train/train.sh train/configs/pi05/libero/pi05_flashvla.yaml`（RoboTwin 配置在 `train/configs/pi05/robotwin/`） |
| 延迟基准 | `python benchmarks/benchmark_latency.py --config_path=benchmarks/configs/latency_flashvla.yaml` |
| 依赖声明 | 构建于 [LeRobot](https://github.com/huggingface/lerobot) 与 [VLASH](https://github.com/mit-han-lab/vlash) |

## 开源边界（截至 2026-08-29）

- **已开源**：训练、仿真评测、延迟基准与部署入口可辨识。
- **权重**：LIBERO / RoboTwin \(\pi_{0.5}\) 检查点已在 Hugging Face 发布。
- **真机数据**：Gello 遥操作演示未随仓发布；论文附录给出任务指令与训练步数。
