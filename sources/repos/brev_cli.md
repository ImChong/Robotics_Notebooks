# brevdev/brev-cli

> 来源归档

- **标题：** NVIDIA Brev CLI
- **类型：** repo / CLI 工具
- **链接：** https://github.com/brevdev/brev-cli
- **产品门户：** https://brev.nvidia.com
- **文档：** https://docs.nvidia.com/brev/latest/
- **许可证：** MIT
- **入库日期：** 2026-09-06
- **一句话说明：** NVIDIA Brev 官方 CLI：在终端创建/管理多云 GPU 实例、`brev login`/`brev create`/`brev ls`，并内置供编码 agent 使用的 **agent-skill**（自然语言管理 GPU）。
- **代码：** https://github.com/brevdev/brev-cli（**已开源** MIT）
- **沉淀到 wiki：** [`wiki/entities/nvidia-brev.md`](../../wiki/entities/nvidia-brev.md)

---

## 安装（README）

| 平台 | 命令 |
|------|------|
| macOS | `brew install brevdev/homebrew-brev/brev` |
| Linux | `curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh \| bash` → `~/.local/bin` |
| Windows | 通过 WSL（Ubuntu ≥22.04）后同上 Linux 安装 |
| Pixi | `pixi global install brev`（conda-forge） |

## 常用命令

```bash
brev login
brev create awesome-gpu-name
brev ls
```

## AI Agent 集成

```bash
brev agent-skill
# 或
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/scripts/install-agent-skill.sh | bash
```

安装后可在 Claude Code 等 agent 中用自然语言创建实例（如「create an A100 for ML training」）。

## 对 wiki 的映射

- [nvidia-brev](../../wiki/entities/nvidia-brev.md)
- [nvidia-physical-ai-learning](../../wiki/entities/nvidia-physical-ai-learning.md) — 课程云 GPU 入口
- [compass](../../wiki/entities/compass.md) — COMPASS 训练可跑在 Brev GPU 实例
