# ZERITH 2.5 官方源码（ZERITH-ZLab/ZERITH-V2.5）

- **类型：** repo
- **仓库：** <https://github.com/ZERITH-ZLab/ZERITH-V2.5>
- **论文：** <https://arxiv.org/abs/2611.23746>
- **研究主页：** <https://www.zerith.com/blog/zerith-2-5/>
- **模型：** <https://huggingface.co/OpenZerith/ZERITH-2.5-LIBERO-Plus>
- **许可：** 代码 Apache-2.0；ZERITH 自有模型权重 CC BY-NC 4.0
- **核对日期：** 2026-11-19
- **核对范围：** 主仓 README 全文、docs/simulation/libero-plus.md、scripts/download_libero_plus_assets.sh、simulation/zerith_sim/posttrain/__main__.py、train.py、serve.py、evaluate.py 和 MODEL_LICENSE_POLICY.md；静态代码阅读，未用 GPU 实际运行。
- **通读：** 已完成
- **映射：** [ZERITH 2.5 独立节点](../../wiki/entities/paper-zerith-v25.md)

## 开源范围

- 提供仿真数据准备、后训练、checkpoint 导出、HTTP 策略服务和 LIBERO / LIBERO-Plus / MetaWorld 评测客户端。
- 官方 Hugging Face 的权重仓需要授权访问；FP32 policy 权重约 **27 GB**，官方建议用 **80 GB GPU** 加载编码器与推理。
- 公开 H1 权重是 **mid-training backbone**，不是一键真机部署策略；还要有经过验证的 H1 inference bundle 和 SDK/IK。

## 官方源码入口逐项核读

| 命令 | 真实调度链路 |
|---|---|
| `python -m zerith_sim.posttrain prepare` | `__main__.py` → `prepare.prepare(args)`，预编码示范 |
| `python -m zerith_sim.posttrain train` | `__main__.py` 构建 recipe，启动 `torch.distributed.run` → `zerith_sim.posttrain.train.main` → dataloader / strategy / Trainer |
| `python -m zerith_sim.posttrain export` | `__main__.py` → `export.export(args)`，导出模型 artifact |
| `python -m zerith_sim.posttrain serve` | `serve.py` 校验 `manifest.json`、`inference.json`，加载策略、Wan VAE / VGGT / 文本编码器，提供 `/health`、`/reset`、`/infer` |
| `python -m zerith_sim.posttrain eval` | `evaluate.py` 创建仿真任务，`PolicyClient` 调模型服务，逐步执行动作并写入 `episodes.jsonl` / `summary.json` |

服务端 `/reset` 重置视觉、几何与 policy 状态并编码文本；`/infer` 从多视图图像和状态构建 latent/geometry，执行 `policy.act_state_trajectory(...)`，只返回前四步动作给客户端，后者按顺序调用模拟器 step。以上是源码逻辑，不是臆测的伪运行流程。

## 官方 LIBERO-Plus 最小闭环

1. 在 Linux + Conda 环境按 README 创建策略推理与仿真两套依赖环境（CUDA 12.8 兼容 PyTorch）。
2. 用 `bash scripts/download_libero_plus_assets.sh` 获取已授权模型、Wan/VGGT、LIBERO-Plus 及基础指令。
3. 从 `simulation/` 启动 `python -m zerith_sim.posttrain serve --artifact assets/policies/libero-plus ... --port 8000`。
4. 第二终端用 `python -m zerith_sim.posttrain eval --bench libero-plus --suite libero_spatial --task-ids 0 30 60 --episodes 1 --server http://127.0.0.1:8000 --output results/libero-plus-quickstart`。
5. 查看 `episodes.jsonl` 与 `summary.json`。该 3-task × 1-episode quickstart 是闭环自检，不是重现论文 93.70% 完整评测。

## 限制与注意

- README 的 Quick Start 克隆示例使用了不同账号的仓库 URL；本归档以官方组织仓库 `ZERITH-ZLab/ZERITH-V2.5` 为入口，复现时须检查 commit、依赖与许可。
- `MODEL_LICENSE_POLICY.md` 明确权重及其改作非商用；代码 Apache-2.0 不覆盖第三方 Wan/VGGT 权重许可。
- 只做静态代码核读，**没有跑通 GPU 推理、论文训练或机器人真机验证**。

## 一手代码位置

- [README](https://github.com/ZERITH-ZLab/ZERITH-V2.5/blob/main/README.md)
- [LIBERO-Plus 指南](https://github.com/ZERITH-ZLab/ZERITH-V2.5/blob/main/docs/simulation/libero-plus.md)
- [CLI](https://github.com/ZERITH-ZLab/ZERITH-V2.5/blob/main/simulation/zerith_sim/posttrain/__main__.py)
- [HTTP 服务](https://github.com/ZERITH-ZLab/ZERITH-V2.5/blob/main/simulation/zerith_sim/posttrain/serve.py)
- [评测实现](https://github.com/ZERITH-ZLab/ZERITH-V2.5/blob/main/simulation/zerith_sim/posttrain/evaluate.py)
