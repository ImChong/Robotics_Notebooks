# WeightCode 官方开源仓库 / Gen-Verse/WeightCode

- **代码：** https://github.com/Gen-Verse/WeightCode
- **论文：** https://arxiv.org/abs/2611.25476
- **项目主页：** https://gen-verse.github.io/WeightCode/
- **数据与会话记录：** https://huggingface.co/datasets/Boom5426/WeightCode
- **开源状态：** 已开源；仓库含实际运行 CLI、模型/程序重放、补丁校验、受限编辑器、Agent Skill 和测试。pyproject 标记 MIT 许可；发表数据需遵循各上游数据集许可。
- **核对日期：** 2026-11-21
- **核对范围：** README.md、README.zh-CN.md、pyproject.toml、skills/weightcode/SKILL.md、weightcode/application/cli.py、application/runtime.py、examples/scan.py、examples/replay.py、patch/executor.py、sandbox/program.py 的可见实现与官方使用步骤；未逐文件审计全部研究实验脚本，也未运行在线下载的复现实验。
- **通读：** 已完成

## CLI 与代码执行入口

- pyproject.toml 注册 `weightcode = weightcode.application.cli:main`；`weightcode reproduce scan` 经 application/cli.py 调度 examples/scan.py。
- scan.py 用 examples/fetch.py 下载/核验 Hugging Face 的网络、会话与已发布程序，以及固定版本 SCAN 文件；重建并校验测试划分。
- examples/replay.py 的 replay_session 调用 harness/program.py 的 run_update，然后 write_derived_patch 与 patch/executor.py 的 apply_patch；严格校验参数哈希、修改边界及重建结果。
- scan.py 最后调用 evalkit/scanskill.py 的 scan_readout_arrays 对照新旧命令，保存 result.json，给出 matches_paper。
- application/cli.py 另提供 import-model、prepare-data、prepare-task、pack-program、apply、synthesize、doctor；application/runtime.py 负责候选评估和只在选择冻结后做最终验证。
- 当前用户侧 Agent Skill 首批支持 FP32 torchvision ResNet-18 state_dict，并要求互不重叠的 experience、selection、final 数据；不能把论文里各种研究网络等同于通用任意架构即插即用。

## 官方最小复现

```bash
git clone https://github.com/Gen-Verse/WeightCode.git
cd WeightCode
uv sync --locked
uv run weightcode reproduce scan
```

- Linux x86-64 与 uv；上述 SCAN 路径按 README 不要求 GPU 或 API Key，但需要下载公开发布的模型、会话及 SCAN 数据。
- 不传沙箱时，`reproduce scan` 核对发布程序 SHA-256 后直接执行该程序；可通过 `--sandbox-python` 指定 bubblewrap 沙箱执行。**不要直接运行来源未知的权重程序**。
- ImageNet 复现还需 `--extra train --extra t03`、Hugging Face ImageNet 使用授权、更多数据下载；README 的默认 500 张评测仅是完整论文 38000 张留出图像的子集，且该命令读取发布的检查点而非重跑原生成程序。
- Agent Skill 的生成程序流程要求 bubblewrap 0.8+，并依赖外部受控代码编辑器；不保证所有编辑被接纳，报告可能是 accepted / base-kept / final-rejected。
- 本次仅核验文档及核心源码路径，**未声称本地复现成功**。

## 对 wiki 的映射

- [WeightCode：权重程序安装新能力](../../wiki/entities/paper-weightcode.md)
