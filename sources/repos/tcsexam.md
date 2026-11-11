# MoonshotAI/TCSExam 官方任务与评测代码

- **仓库：** https://github.com/MoonshotAI/TCSExam （官方 master）
- **论文 v1：** https://arxiv.org/abs/2611.07962v1
- **项目页：** https://moonshotai.github.io/TCSExam/
- **论文归档：** [TCSExam](../papers/tcsexam_arxiv_2611_07962.md)
- **站点归档：** [TCSExam 项目](../sites/tcsexam.md)
- **wiki：** [TCSExam 实体](../../wiki/entities/paper-tcsexam.md)
- **核对时间：** 2026-11-11
- **开源：** 任务/框架、验证器与 CLI 可读取；公开源码不等于包含能离线还原论文每个受限资源或隐藏 backend 的全部凭证。
- **许可证：** 作者项目代码 Apache-2.0；原始任务陈述、官网/文档 CC BY 4.0；PDF 单独 CC BY-NC-ND 4.0；第三方按 docs/licenses/ 逐项审核。

## 源码文件与职责（按官方 README + 静态源码核对）

| 路径 | 职责 |
|---|---|
| pyproject.toml | 安装 tcsexam CLI，依赖 Python >=3.12；runtime 使用 Harbor 0.22.0 |
| src/tcsexam/__main__.py | python -m tcsexam 入口，转到 cli.main |
| src/tcsexam/cli.py | list/build/setup/prepare/doctor/run/score/solver-score 等命令分发 |
| src/tcsexam/layout.py | 从 suites/ 清单定位 task、solver 变体与路径 |
| src/tcsexam/setup.py、build.py | 核查/准备依赖资源、Docker 镜像与缓存 |
| src/tcsexam/diagnostics.py | doctor 检查宿主环境、任务资源、agent 配置和 backend 条件 |
| src/tcsexam/runtime.py | materialize：复制任务包、注入固定镜像/运行配置和 provenance |
| src/tcsexam/harbor_launch.py | 通过 Harbor 运行 agent 及分离的验证阶段 |
| src/tcsexam/scoring.py | aggregate 汇总四赛道，relative_score 计算固定参考 Solver 分 |
| tasks/、suites/ | 任务材料与 Full/Lite 选择清单；30 个 Solver 题有 Online/Offline 两套目录 |

## 实际执行链（已静态对照）

1. `tcsexam run` 进入 `src/tcsexam/cli.py:main`，解析 suite / task / solver-protocol。
2. `configure_agent` 检查模型端点凭证；`resolve` 用 `layout.selection` 选择任务。
3. `setup_selection` 确认镜像与资源；`doctor` / `diagnostics.prerequisites` 检查 ready 状态。
4. `prepare` 经 `runtime.materialize` 生成隔离的 Harbor task copies 与 JSON JobConfig；再以 `python -m tcsexam.harbor_launch run --config ...` 启动任务。
5. 提交/verification 结果由各任务定义的 verifier 处理；`tcsexam score` 调用 `scoring.aggregate` 离线汇总。不要说打分在 `run` 里自动完成或所有赛道共用同一 verifier。
6. `solver-score` 采用固定参考表计算相对分值，`scoring.aggregate` 禁止混合 Solver Online/Offline 尝试；未知/缺失验证保留 null。

## 官方最小上手

```bash
git clone https://github.com/MoonshotAI/TCSExam.git
cd TCSExam
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,backend]'
tcsexam validate
tcsexam list --suite lite
tcsexam source-inputs
tcsexam build --suite lite --task discrete-bidding-games
tcsexam doctor --suite lite --task discrete-bidding-games
# 配置 .env 并显式 source 后，才运行实际 agent / 模型 API
tcsexam run --suite lite --task discrete-bidding-games
```

官方要求 Linux x86-64、Python 3.12+、Docker Engine + Compose + Buildx、模型 API。Solver 额外依赖 Landlock ABI >=3 与容器 syscall 配置；部分 Online 运行依赖独立 judge backend。上述命令摘自官方 getting-started；**本次没有在 Docker 主机执行**。

## 对复现的影响

- Full 默认 Online Solver、Lite 默认 Offline Solver：必须显式记录。
- 只改 suite 不等于公平消除题目难度、工具设置、预算与 API 差别。
- verifier 与 development 隔离，缺资源/缺签名/缺网络授权时应判为 blocked，不能擅自标成功或零分。

## 官方一手来源

- [README](https://github.com/MoonshotAI/TCSExam/blob/master/README.md)
- [运行入门](https://github.com/MoonshotAI/TCSExam/blob/master/docs/getting-started.md)
- [评分规则](https://github.com/MoonshotAI/TCSExam/blob/master/docs/scoring.md)
- [源码](https://github.com/MoonshotAI/TCSExam/tree/master/src/tcsexam)
