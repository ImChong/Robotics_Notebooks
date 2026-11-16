# Apple AMP（AMP1 / AMP1+）官方开源实现

> 官方代码仓来源归档（不是第三方转载）。

- **类型：** repo
- **仓库：** <https://github.com/apple-aiml-research/ml-amp>
- **项目页：** <https://apple-aiml-research.github.io/ml-amp/>
- **论文：** <https://arxiv.org/abs/2611.09892>
- **核验日期：** 2026-11-10
- **核对范围：** README、`pyproject.toml`、`docs/quickstart.md`、`docs/reproducibility.md`、`docs/reproducing-the-paper.md`，以及 `src/amp/cli.py`、`src/amp/eval/robolab_official.py`、`src/amp/agents/run.py`、`src/amp/agents/loop.py` 关键代码路径和 LICENSE。
- **通读：** 已完成

## 开源状态与源码入口

- **已开源，可辨识可运行入口：** `pyproject.toml` 注册 `amp = "amp.cli:main"`；`src/amp/cli.py` 将 `amp eval robolab` 映射到 `amp.eval.robolab_official:main`，其他入口含 `capgym`、`robodojo`、`practice` 和 `doctor`。
- **RoboLab 运行链：** `src/amp/eval/robolab_official.py` 管理任务、slot/GPU 和仿真服务；`src/amp/agents/run.py` 读取 episode 配置、组装提示词和 SDK、建立沙箱 Kernel 并启动 `Agent.run()`；`src/amp/agents/loop.py` 维护多轮模型/工具循环。
- **执行接口：** 持久 Python / SDK 与机器人服务交互（观测、感知、运动），模型 API 可使用 Anthropic、Gemini 或兼容服务。发布了 `docs/reproducing-the-paper.md` 对应实验的逐项命令。
- **许可：** 官方 `LICENSE` 是 Apple 自有软件许可条款，其他组件参见 `ACKNOWLEDGEMENTS`；不可简单称“全部 MIT 许可”。

## 官方复现（Linux x86_64 + NVIDIA GPU）

```bash
git clone https://github.com/apple-aiml-research/ml-amp AMP && cd AMP
git clone https://github.com/NVlabs/RoboLab.git external/robolab
git -C external/robolab checkout 9db0aaf09d9fe5d4f37b168320788258c7012463
git -C external/robolab lfs pull
uv sync --frozen
export GEMINI_API_KEY=...
uv run amp doctor
uv run amp eval robolab --harness amp1_basic --model gemini-4-argon \
  --reasoning high --task-set benchmark --gpus auto --output runs/robolab
```

- **论文实验环境：** Debian 13 / H100 80GB / NVIDIA 580.159.03；RoboLab 使用 Python 3.11、Isaac Sim 5.0.0 / Isaac Lab 2.2.0。
- **第三方 pin（参考 `docs/reproducibility.md`）：** RoboLab `9db0aaf09d9fe5d4f37b168320788258c7012463`；CaP-X `53e9966d7a8e2fa7494676772bccc35280f5c0ed`；RoboDojo `09bd5f5e762b949f5858ece31347ce62057b3a5e`。
- **复现边界：** 需模型密钥或服务、第三方代码与资产、相应许可证；文档明确不同 GPU/driver/系统可能令实验数值略有变化。此归档未执行上述命令。

## 关联资料

- [论文原文归档](../papers/apple_amp1_arxiv_2611_09892.md)
- [官方项目页归档](../sites/apple-aiml-research-ml-amp.md)
- [统一 wiki 项目节点](../../wiki/entities/paper-apple-amp1.md)
