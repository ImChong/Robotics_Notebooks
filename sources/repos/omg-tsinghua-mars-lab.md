# tsinghua-mars-lab / OMG

> 来源归档

- **标题：** OMG: Omni-Modal Motion Generation for Generalist Humanoid Control
- **类型：** repo（训练 / 推理 / 评测 / G1 部署代码）
- **维护方：** Tsinghua University MARS Lab
- **仓库：** <https://github.com/tsinghua-mars-lab/OMG>
- **项目页：** <https://tsinghua-mars-lab.github.io/OMG/>
- **License：** MIT
- **入库日期：** 2026-06-10
- **一句话说明：** 清华 MARS Lab 发布的 **omni-modal 人形运动生成** 开源栈：OMG-DiT 扩散训练与 ONNX/TensorRT 推理、HoloMotion tracker 耦合管线、benchmark 与 **G1 实时部署**（GPU 工作站 planner + Orin bridge）；OMG-Data / 预训练权重 HF 发布标注 coming soon。
- **沉淀到 wiki：** [`wiki/entities/paper-omg-omni-modal-humanoid-control.md`](../../wiki/entities/paper-omg-omni-modal-humanoid-control.md)

---

## 发布状态（README 勾选，截至入库日）

| 组件 | 状态 |
|------|------|
| Training Code | ✅ |
| Inference Code | ✅ |
| Benchmark Code | ✅ |
| Sim-to-Real Deployment Code | ✅ |
| Evaluation Pipelines / Evaluator CKPT | ⏳ |
| OMG-Data | ⏳（HF coming soon） |
| Pretrained Checkpoints | ⏳ |

---

## 端到端工作流（仓库归纳）

1. `make venv` + `make install`（大陆网络可用 `make install-cn`）
2. 下载 OMG-Data、生成器 checkpoint、**HoloMotion** tracker ONNX（官方不随仓分发，需从 [HoloMotion](https://github.com/HorizonRobotics/HoloMotion) / [HF](https://huggingface.co/HorizonRobotics/HoloMotion_models) 获取）
3. 可选 `scripts/materialize_omg_data.sh` 预计算训练分片
4. `compute_stats` → `omg.cli.generation.train`（exp 50m–1b）
5. `export_onnx` → TensorRT/CUDA 推理
6. `omg.cli.pipeline.main`（`diffusion-only` / `tracker-only` / `sync` / `async` / `offline-track`）
7. benchmark（需 OMG-Evaluator，HF 待发布）
8. **G1 真机**：HoloMotion 部署于 Orin + OMG realtime planner（GPU 工作站）+ real bridge；真机推荐 **velocity_tracking** ONNX

### 推荐本地目录布局

```text
data/OMG-Data/omg_data/ + materialized/
models/generation/ + evaluator/ + t5-base-local/ + holomotion/
```

环境变量：`OMG_DATA_ROOT`、`OMG_MATERIALIZED_ROOT`、`OMG_MODELS_ROOT`。

### 模型规模配置

`configs/generation/exp/`：**50m / 100m / 300m / 500m / 1b**。

文本条件默认 `t5-base`（`${OMG_MODELS_ROOT}/t5-base-local` 或 HF `google-t5/t5-base`）。

---

## 与 HoloMotion 的关系

OMG **不重新分发** HoloMotion 权重；tracker 层复用 [HorizonRobotics/HoloMotion](https://github.com/HorizonRobotics/HoloMotion) 的 **motion_tracking** 与 **velocity_tracking** ONNX。知识库实体见 [HoloMotion](../../wiki/entities/holomotion.md)。

---

## 对 wiki 的映射

- 项目页：[sources/sites/omg-tsinghua-mars-lab-github-io.md](../sites/omg-tsinghua-mars-lab-github-io.md)
- 实体页：[wiki/entities/paper-omg-omni-modal-humanoid-control.md](../../wiki/entities/paper-omg-omni-modal-humanoid-control.md)
- 方法交叉：[扩散运动生成](../../wiki/methods/diffusion-motion-generation.md)、[ETH G1 扩散全身 locomotion](../../wiki/entities/paper-hrl-stack-27-learning_whole_body_humanoid_locomot.md)、[Heracles](../../wiki/entities/paper-heracles-humanoid-diffusion.md)
