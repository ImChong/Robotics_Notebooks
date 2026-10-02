# dawn-parkour：DAWN 官方实现

- **类型：** repo
- **代码：** <https://github.com/DocyNoah/dawn-parkour>
- **项目页：** <https://dawn-parkour.github.io/>；[项目页归档](../sites/dawn-parkour.md)。
- **论文：** <https://arxiv.org/abs/2609.29092>；[论文归档](../papers/dawn_arxiv_2609_29092.md)。
- **核查日期：** 2026-10-02
- **许可证：** MIT。
- **开源状态：** 训练与仿真回放源码已发布；README 使用本地训练 checkpoint，未列独立权重下载；目录未见 Go1 SDK / Jetson 真机通信入口。

## 可辨识入口

| 文件 / 模块 | 用途 |
|---|---|
| `INSTALL.md`、`setup_isaaclab_env.sh` | Isaac Sim 5.1.0、固定 IsaacLab 子模块与 uv 安装 |
| `dawn/train.py` | 启动模拟器、创建环境与 runner，调用训练 |
| `dawn/rl/runners/dawn_runner.py` | rollout、世界模型特征、动作历史与 PPO/AMP 更新 |
| `dawn/rl/modules/dawn_world_model.py` | `image_clean` 重建目标、双向交叉熵对比损失 |
| `dawn/sim/utils/depth_noise_model.py` | 深度噪声实现 |
| `dawn/play.py`、`dawn/rl/runners/dawn_eval_runner.py` | checkpoint 仿真回放 |
| `dawn/datasets/mocap_motions/` | hop/trot 动作文件 |

README 使用 RTX 5090、默认 4,096 环境；不能承诺 8 GB 显存直接运行默认配置。本次仅静态核查源码。

## 复现入口

按 `INSTALL.md` 初始化子模块、安装 Isaac Sim、执行 `uv sync --extra cu128` 与 setup 脚本后：

```bash
source .venv/bin/activate
python -m dawn.train dawn
python -m dawn.play --checkpoint.model-dir "logs/DAWN/<run>/model_<iteration>" --env.task Go1-DAWN-Play-v0 --env.num-envs 4 --video.enable
```

checkpoint 路径需替换为实际训练输出。

**知识页：** [DAWN](../../wiki/entities/paper-dawn.md)。
