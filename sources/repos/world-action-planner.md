# world-action-planner（XiangchengZhang）

> 来源归档

- **标题：** World Action Planner — action-conditioned world models + planning demo
- **类型：** repo（世界模型训练/服务 + 仿真环境 + 想象动作 notebook）
- **来源：** Harvard University（Xiangcheng Zhang / Yilun Du）
- **链接：** <https://github.com/XiangchengZhang/world-action-planner>
- **项目页：** <https://worldactionplanner.github.io/>
- **论文：** <https://arxiv.org/abs/2607.27599>
- **权重：** <https://huggingface.co/XiangchengZhang/world-action-planner>
- **入库日期：** 2026-08-02
- **一句话说明：** 官方实现：Wan 系 pose-image 条件世界模型服务、LIBERO/robosuite 环境包、`demo.ipynb` 想象 rollout；权重走 Hugging Face。
- **开源状态：** **已开源**（2026-08-02）；含 `world_model/`、`wm_client/`、`environments/`、`start_server.sh`。
- **沉淀到 wiki：** [`wiki/entities/paper-world-action-planner.md`](../../wiki/entities/paper-world-action-planner.md)

## 仓库概况（2026-08-02）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`XiangchengZhang/world-action-planner`） |
| 世界模型 | `world_model/`（独立 env；`server.py` WebSocket；Hydra） |
| 客户端 | `wm_client/`（`WMClient` / `WMEnv`） |
| 仿真 | `environments/{robomimic,robosuite,LIBERO}` editable install |
| Demo | `demo.ipynb`（需先 `bash start_server.sh`） |
| 架构声明 | 基于 Large Video Planner + Diffusion Forcing Transformer |
| 许可 | 以仓库 LICENSE 为准（入库时 README 未强调专有限制） |

## README 课程映射

| 路径 | 内容 |
|------|------|
| `world_model/README.md` | mamba env、HF 下载 ckpt、Wan VAE、启动 `server.py` |
| `start_server.sh` | 一键起服务（默认端口 7880） |
| `demo.ipynb` | LIBERO 任务 + 想象动作视频导出 |
| `wm_client/` | WebSocket 客户端与 `WMEnv` 包装 |
| `environments/` | robomimic / robosuite / LIBERO 本地包 |

### 权重下载（README）

```bash
huggingface-cli download XiangchengZhang/world-action-planner \
  --include "world_models/**" \
  --local-dir data/ckpts \
  --local-dir-use-symlinks False
```

默认示例：`libero_90_base/checkpoints/latest.ckpt`；另有 `libero_object_ft`、`robosuite_ft` 与 DP/IDM ckpt。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | [`paper-world-action-planner.md`](../../wiki/entities/paper-world-action-planner.md) |
| 项目页 | [`worldactionplanner-github-io.md`](../sites/worldactionplanner-github-io.md) |
| 权重站 | [`huggingface-xiangchengzhang-world-action-planner.md`](../sites/huggingface-xiangchengzhang-world-action-planner.md) |
| 论文源 | [`world_action_planner_arxiv_2607_27599.md`](../papers/world_action_planner_arxiv_2607_27599.md) |
| 基准 | [`libero-benchmark.md`](../../wiki/entities/libero-benchmark.md) |
