# Xiaomi-Robotics-1（官方代码仓：后训练、推理与基准评测）

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-1 — Code for Xiaomi-Robotics-1
- **类型：** repo（官方 GitHub，Apache-2.0；2026-08-03 发布代码与 5B/评测权重）
- **组织：** Xiaomi Robotics
- **代码：** <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>
- **Hugging Face 合集：** <https://huggingface.co/collections/XiaomiRobotics/xiaomi-robotics-1>
- **ModelScope 合集：** <https://www.modelscope.cn/collections/XiaomiRobotics/Xiaomi-Robotics-1>
- **项目页：** <https://robotics.xiaomi.com/xiaomi-robotics-1.html>
- **论文：** <https://arxiv.org/abs/2607.15330>
- **入库日期：** 2026-07-21（2026-11-14 实读仓库全量代码并更新）
- **核对范围：** 2026-11-14 通读仓库核心目录与入口源码：`README.md`、`LICENSE`（Apache-2.0）、`docs/DEPLOYMENT.md`、`scripts/deploy.sh`、`deploy/server.py`、`deploy/client.py`、`eval_robocasa/{README.md,main.py}`、`eval_robocasa365/{README.md,entry.py}`、`eval_vlabench/{README.md,main.py}`、`xr1/{README.md,docs/data_format.md,scripts/train.sh,scripts/deploy.sh,configs/config.yaml,configs/model/posttrain.yaml,configs/data/load_washer.yaml,tools/train.py,tools/weight_convert.py,tools/compute_normalize.py}` 及 `xr1/mibot/{models/VLA/XR1.py,models/runner/base_runner.py,server/deploy.py,server/runtime/server.py,server/runtime/client.py}`。
- **通读：** 已完成
- **一句话说明：** 2026-08-03 发布可执行代码与 5B 基座/评测权重（Apache-2.0），包含 HF `AutoModel` Socket 推理服务（`deploy/`）、RoboCasa / RoboCasa365 / VLABench 评测入口（`eval_*/`）以及基于 Hydra + Lightning + DeepSpeed 的 5B 后训练与异步部署栈（`xr1/`）；100k h UMI 预训练流水线、Qwen3.5 自动标注脚本及 2.6B/10.5B 权重未公开。

## 开放核查（步骤 2.5，2026-11-14 实测代码仓）

| 类别 | 状态 | 说明 |
|------|------|------|
| GitHub 仓库与许可 | **已开源（Apache-2.0）** | `XiaomiRobotics/Xiaomi-Robotics-1`（2026-08-03 发布 `xr1/`、`deploy/`、`eval_*`） |
| 通用推理与评测入口 | **已发布** | `scripts/deploy.sh` → `deploy/server.py`（`transformers==4.57.1` + `flash-attn==2.8.3`）；`eval_robocasa/main.py`、`eval_robocasa365/entry.py`、`eval_vlabench/main.py` 走 `deploy/client.py` Socket 推理 |
| 后训练与异步部署栈（`xr1/`） | **已发布（仅 5B）** | `tools/weight_convert.py` + `tools/compute_normalize.py` + `scripts/train.sh` (`tools/train.py` → `BaseRunner` → `XR1`) + `xr1/scripts/deploy.sh` (`mibot/server/deploy.py`)；默认 `async_train: true`，`loss = 0.5*loss_mse + freq_coeff*loss_freq + 0.5*loss_l1 + 0.5*loss_score`（FFT 频域损失排除基座速度维 `[17,18,19]`） |
| 模型权重与演示数据 | **已发布 5B 与三基准权重** | HF/ModelScope 发布 `Xiaomi-Robotics-1-5B`、`-{RoboCasa,RoboCasa365,VLABench}` 与 `xr1_post_train_demo`；RoboDojo 未单列权重；**2.6B / 10.5B 未发布** |
| UMI 预训练与自动标注 | **未公开** | 100k+ h UMI 预训练脚本、Qwen3.5-27B 自动标注管线与 VLM NTP 混训代码不在仓内（`XR1.forward()` 直接 `pop("labels")`）；>7.2k h 自采真机集未公开 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) | 实体页；已据此绘制 `## 源码运行时序图` 并明确「后训练开源 / 预训练闭源」边界 |
| [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) | 同组织开源 VLA；共源 `mibot` / Hydra + Lightning + Socket 异步部署工程范式 |
| [VLA](../../wiki/methods/vla.md) | 100k h UMI scaling 基座策略与 5B 后训练开源实现 |

## 对 wiki 的映射

- 更新 **`wiki/entities/xiaomi-robotics-1.md`**：补全 `code:` 字段、源码架构与 `## 源码运行时序图`、`## 结论` 与评测口径注记。
