# RPent（Recursive Physical Agent / Harness VLA 官方实现）

> 来源归档

- **标题：** RPent — Agentic Infrastructure for the Physical World
- **类型：** repo
- **组织：** RLinf（与 [RLinf](rlinf.md) 训练系统同生态）
- **代码：** <https://github.com/RLinf/RPent>
- **文档：** <https://rpent.readthedocs.io/en/latest/>
- **论文：** <https://arxiv.org/abs/2607.08448>（Harness VLA）
- **项目页：** <https://harnessvla.github.io/>
- **HF：** <https://huggingface.co/RLinf>（含 `rlinf-pi05-libero-130-fullshot-sft` 等）
- **入库日期：** 2026-07-22
- **一句话说明：** Harness VLA 的官方 **agentic 运行时**：LLM planner（Claude Code / Codex / API）经 JSON 原语接口调用冻结 VLA（π₀.₅ 等）与解析控制器；默认栈对接 RLinf runtime + openpi + LIBERO-Pro。

## 开源状态（项目页 / README 核查 2026-07-22）

- **已开源：** Python 包 `rpent`（`rpent.cli.main:main`）、planner 模块（`claude_code` / `codex` / `api_loop`）、`rpent/tools` 原语工具包、`resources/libero/memory/` 任务记忆与结果样例、LIBERO-Pro / RoboCasa 安装与运行脚本。
- **依赖栈：** `.[full]` = rlinf + openpi + libero-pro；需自备 Anthropic/OpenAI API key 与 π₀.₅ checkpoint。
- **成熟度：** README 标 Pre-Alpha；Feature Matrix 中 Pi0.5 + LIBERO-PRO 已勾选，RoboCasa / 真机条目部分待完善。

## 快速入口（对齐 README）

```bash
git clone https://github.com/RLinf/RPent rpent && cd rpent
pip install -e ".[full]"
export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
export LIBERO_TYPE=pro
rpent --suite libero_object_swap --task 2 --seed 0 \
  --planner api --model anthropic:claude-opus-4-8 --max-tokens 8192
```

RoboCasa：`bash scripts/setup_robocasa.sh` → `bash scripts/run_robocasa.sh <task> <gpu> <seed>`。

## 对 wiki 的映射

- [Harness VLA（论文实体）](../../wiki/entities/paper-harness-vla.md)
- [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)
- [RLinf 训练系统归档](rlinf.md) — 底层 runtime / 权重生态，≠ 本 agent 仓
