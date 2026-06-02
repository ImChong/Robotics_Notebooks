# MobileGym（Purewhiter/mobilegym）

> 来源归档

- **标题：** MobileGym
- **类型：** repo + 官方文档 + 在线 Demo
- **组织：** 中科院自动化所等（论文作者团队维护）
- **代码：** <https://github.com/Purewhiter/mobilegym>
- **官网 / Live Demo：** <https://mobilegym.dev>
- **论文：** <https://arxiv.org/abs/2605.26114>
- **项目页：** <https://mobilegym.github.io/>
- **许可：** 代码 Apache-2.0；`mobilegym-data` 与合成内容 CC BY-NC 4.0（见 `LICENSE-DATA`、`DISCLAIMER.md`）
- **入库日期：** 2026-06-02
- **一句话说明：** 浏览器内 Android 式移动仿真 + `bench_env` 评测/RL 运行时：28 App、416 任务模板、确定性状态裁判、Playwright 并行 rollout 与多 Agent 适配器。
- **沉淀到 wiki：** [MobileGym](../../wiki/entities/mobilegym.md)

---

## 仓库结构（维护索引）

| 路径 | 作用 |
|------|------|
| `os/` | SystemShell、TaskManager、Intent、通知/权限等服务 |
| `apps/` | 12 个日常 App（微信、支付宝、小红书、B 站、12306 等研究替身） |
| `system/` | 16 个系统 App（Launcher、Settings、AnswerSheet 等） |
| `bench_env/` | Python + Playwright：任务模板、judge、reward、并行 runner、Agent 适配器 |
| `bench_env/task/` | 按 suite 组织（`wechat/`、`crossapp_*` 等） |
| `bench_env/agent/` | autoglm、uitars、venus、gui_owl、generic、human 等 |
| `bench_env/splits/` | test / train 等划分 |
| `docs/platform/` | App 模块契约、状态模型、runtime API |
| `scripts/server/` | nginx gateway（大规模并行 / RL，`https://localhost:4180`） |
| `mobilegym-data/` | 可替换默认 JSON 数据（需单独下载 release） |

---

## Quick Start（摘自 README）

```bash
git clone https://github.com/Purewhiter/mobilegym.git
cd mobilegym && npm install
pip install -r bench_env/requirements.txt && playwright install chromium
# 可选：下载 mobilegym-data-v1.tar.gz（~1.4 GB）
```

**服务模式：**

| 场景 | 命令 | URL |
|------|------|-----|
| 手玩 / 开发 | `npm run dev` | `http://localhost:3000` |
| 单 Agent 评测（≤8 并行） | `npm run build && npm run preview -- --port 4173` | `http://localhost:4173` |
| 大规模 benchmark / RL | `./scripts/server/start_nginx_gateway.sh` | `https://localhost:4180` |

**评测示例：**

```bash
python -m bench_env.run --list
python -m bench_env.run --split test --parallel 8 \
  --env-url http://localhost:4173 \
  --agent autoglm --model-name autoglm-phone-9b
```

---

## Agent 适配器（README 表）

| `--agent` | 说明 |
|-----------|------|
| `autoglm` | Open-AutoGLM 中文 prompt；对标 AutoGLM-Phone-9B |
| `uitars` | UI-TARS-1.5-8B |
| `venus` | UI-Venus-1.5-8B |
| `gui_owl` | GUI-Owl-1.5-Think |
| `generic` / `generic_v2` | 统一 JSON；支持 RL checkpoint |
| `mai_ui` | MAI-UI 风格 |
| `human` | 人工调试 judge |

新增适配器：~100 行 + 注册 `bench_env/agent/__init__.py`。

---

## 扩展要点

- **新 App：** `apps/MyApp/` 或 `system/` — `manifest.ts`、`navigation.declaration.ts`、`data/defaults.json`；**manifest 自动发现**，不改 OS/benchmark 层。
- **新任务：** `bench_env/task/<suite>/` — `description`、`setup`（JSON 注入）、`check_goals()` / `get_answer()`；见 `TASK_AUTHORING_GUIDE.md`。
- **路线图（README）：** 训练代码（Online RL pipeline）尚未发布。

---

## 对 wiki 的映射

- 主实体：[MobileGym](../../wiki/entities/mobilegym.md)
- 论文摘录：[mobilegym_arxiv_2605_26114.md](../papers/mobilegym_arxiv_2605_26114.md)
- 官网归档：[mobilegym-dev.md](../sites/mobilegym-dev.md)

## 外部参考

- [Purewhiter/mobilegym](https://github.com/Purewhiter/mobilegym)
- [mobilegym.dev](https://mobilegym.dev)
- [arXiv:2605.26114](https://arxiv.org/abs/2605.26114)
