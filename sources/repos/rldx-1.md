# RLDX-1

> 来源归档

- **标题：** RLDX-1 Technical Report（仓库 README 与论文同源）
- **类型：** repo + technical report（arXiv）
- **组织：** RLWRLD
- **链接：** https://github.com/RLWRLD/RLDX-1
- **论文：** https://arxiv.org/abs/2605.03269
- **项目页：** https://rlwrld.ai/rldx-1
- **模型集合：** https://huggingface.co/collections/RLWRLD/rldx-1
- **Stars / Forks：** ~131 / 0（2026-05，以 GitHub 页面为准）
- **入库日期：** 2026-05-11
- **一句话说明：** 面向类人灵巧操作的 Vision-Language-Action（VLA）开源实现，在 Qwen3-VL 与 GR00T N1.7 训练范式之上引入 MSAT 动作头、运动感知 / 长时记忆 / 物理传感三能力与三阶段训练，并配套图捕获与 RTC 的低延迟推理栈。
- **沉淀到 wiki：** [RLDX-1](../../wiki/entities/rldx-1.md)

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | RLDX-1 是带多流扩散动作头与可选物理模态的 VLA 工程参考 |
| [LeRobot](./lerobot.md) | 数据格式约定为 LeRobot v2.1 + `meta/modality.json` |
| NVIDIA GR00T N1.7 | 代码在 GR00T N1.7 框架上扩展；EmbodimentTag 与 per-embodiment MLP 头约定一致 |
| Qwen3-VL / FLUX | 视觉–语言骨干与 MM-DiT 向动作建模的扩展（MSAT） |

---

## 核心能力（README 归纳）

1. **Multi-Stream Action Transformer（MSAT）**  
   认知、物理（触觉 / 力矩等）、动作各一条流，经联合 self-attention 耦合；叙述上为 MM-DiT 向动作建模的扩展。

2. **运动感知（Motion awareness）**  
   多帧观测 + motion 模块；中间层对视频 token 做压缩以控制策略侧算量。

3. **长时记忆（Long-term memory）**  
   记忆模块融合历史认知特征与当前特征，超出短多帧窗口。

4. **物理传感（Physical sensing）**  
   触觉与力矩等进入独立 physics stream；解码器联合预测未来物理信号（训练目标之一）。

5. **三阶段训练**  
   Pre-training（泛化）→ Mid-training（能力注入，如 DROID / ALLEX）→ Post-training（任务适配）；合成数据增强稀有操作场景。

6. **实时推理**  
   静态图捕获 + 自定义融合算子；README 给出 RTX 5090 上约 **43.7 ms/step**、**>22 Hz** 的量级（与具体 `--compile` / RTC 配置相关）。

---

## 工程要点（便于映射到 wiki）

| 维度 | 要点 |
|------|------|
| 环境 | Python 3.10、CUDA 12.x、`uv` 包管理 |
| 数据 | LeRobot v2.1；`EmbodimentTag` 选择 per-robot MLP 槽位 |
| 训练 CLI | `rldx/experiment/launch_train.py`，`--use-memory` / `--use-motion` / `--use-physics` 等开关 |
| 推理 | `RLDXPolicy` 进程内；`run_rldx_server.py` ZeroMQ；`--compile`、`--rtc-inference-mode` |
| 许可证 | 代码 Apache 2.0；权重为 RLWRLD 非商用许可（见 HF 上各 checkpoint） |

---

## 对 wiki 的映射

- **实体页**：[RLDX-1](../../wiki/entities/rldx-1.md) — 架构、训练阶段、数据与推理栈的压缩说明与交叉引用。
- **方法总览**：[VLA](../../wiki/methods/vla.md) — 将 RLDX-1 作为「多模态动作扩散 + 触觉 / 力矩条件 + 工程实时优化」路线的代表之一。

---

## BibTeX（仓库 README）

```bibtex
@article{rldx2026,
  title={RLDX-1 Technical Report},
  author={Dongyoung Kim and Huiwon Jang and Myungkyu Koo and Suhyeok Jang and Taeyoung Kim and others},
  year={2026},
  journal={arXiv preprint arXiv:2605.03269},
  eprint={2605.03269},
  archivePrefix={arXiv}
}
```
