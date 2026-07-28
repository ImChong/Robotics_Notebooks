# Source: MimicKit (xbpeng/MimicKit)

- **Title**: MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **URL**: https://github.com/xbpeng/MimicKit
- **Project page**: https://motion.stanford.edu/research/mimickit
- **Paper (Starter Guide)**: https://arxiv.org/abs/2510.13794
- **Author**: Xue Bin Peng
- **Year**: 2025
- **Type**: Codebase / Research Framework
- **License**: Apache-2.0
- **规模**：约 2.2k stars（2026-07 核查），最近推送 2026-06-23，主分支 `main`

## 核心内容（仓库 README 归纳）

- **定位**：轻量级、模块化的强化学习框架，专为物理机器人/角色的运动模仿与控制设计；强调「clean and lightweight, minimal dependencies」。更大规模、更全栈的姊妹框架为 [ProtoMotions](https://github.com/NVlabs/ProtoMotions/)。
- **集成算法**：DeepMimic, AMP, AWR, ASE, LCP, ADD, SMP（每个算法在 `docs/README_<算法>.md` 有独立说明）。

### 多仿真后端（Engines）

| 后端 | 用法 | 版本基线（README 标注） |
|------|------|------------------------|
| Isaac Gym | `--engine_config data/engines/isaac_gym_engine.yaml` | 官方 Isaac Gym 发行版 |
| Isaac Lab | `--engine_config data/engines/isaac_lab_engine.yaml` | 测试过 commit `2ed331acfcbb1b96c47b190564476511836c3754` |
| Newton | `--engine_config data/engines/newton_engine.yaml` | 测试过 `v1.0.0` |

官方建议为每个仿真器用 Conda 建独立 Python 环境；安装后 `pip install -r requirements.txt`，再从作者提供的 SharePoint 打包下载 assets 与 motion data 解压到 `data/`。

### 训练 / 测试 / 分布式

- 训练入口：`python mimickit/run.py --mode train --num_envs 4096 --engine_config ... --env_config ... --agent_config ... --visualize true --out_dir output/`。
- 参数可收敛到 `arg_file`（如 `args/deepmimic_humanoid_ppo_args.txt`），与命令行等价；全算法预置参数在 `args/`。
- 测试：`--mode test --model_file <pt>`；**预训练模型**在 `data/models/`，对应训练日志在 `data/logs/`。
- **分布式训练**：`--devices cuda:0 cuda:1 ...` 支持多 CPU / 多 GPU 多进程并行。
- **日志**：`--logger` 支持 `txt` / TensorBoard `tb` / `wandb`；`--video true` 可无头录制视频；`tools/plot_log/plot_log.py` 可绘制 `log.txt` 曲线。
- 可视化交互：`Alt`+左键平移相机、滚轮缩放、`Enter` 暂停、`Space` 单步。

### 动作数据与重定向

- 动作以 `.pkl` 存储（`mimickit/anim/motion.py` 的 `Motion` 类）：每帧为 `[root position (3D), root rotation (3D), joint rotations]`，3D 旋转用 **3D 指数映射**（Exponential Maps），关节顺序为运动学树深度优先遍历。
- `motion_file` 既可指单条 clip，也可指 `data/datasets/` 下的 **数据集文件**（多 clip 混合训练）。
- `view_motion` 环境可回放动作 clip 做数据检查。
- **重定向工具链**：
 - `tools/gmr_to_mimickit/` — 将 [GMR](https://github.com/YanjieZe/GMR) 输出转为 MimicKit 格式；
 - `tools/smpl_to_mimickit/` — 将 [AMASS](https://amass.is.tue.mpg.de/) 的 SMPL 动作转为 MimicKit 格式。

## 与 ProtoMotions 的关系

- **MimicKit**：偏算法与论文复现的轻量框架，运动模仿方法族谱集中、依赖少。
- **ProtoMotions**：偏大规模并行仿真、多后端、数据管线与部署导出的全栈研究平台；README 自述「更 feature-rich」。

## BibTeX（仓库提供）

```bibtex
@article{MimicKitPeng2025,
  title={MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control},
  author={Peng, Xue Bin},
  year={2025},
  eprint={2510.13794},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2510.13794},
}
```
