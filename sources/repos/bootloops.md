# BootLoops 1.0

- **类型：** repo
- **维护者：** Matthew D. Schwartz
- **代码：** https://github.com/BootLoops-ai/bootloops
- **核查版本：** 66b680ce742e654cfe86da4f072a69061fe182b1
- **固定版本：** https://github.com/BootLoops-ai/bootloops/tree/66b680ce742e654cfe86da4f072a69061fe182b1
- **项目页：** https://www.bootloops.ai（本次未成功读取）
- **入库日期：** 2026-10-02
- **开源状态：** 通用 toolkit 与自有引擎源码已发布；不是所有文章成果的一站式复现包。
- **许可：** 主体 MIT；仓内文字/图 CC BY 4.0；若干文件及外部引擎有独立 GPL 等许可，见 THIRD_PARTY.md。
- **交叉归档：** [客座研究文章](../sites/anthropic-claude-shaped-science.md)
- **沉淀到 wiki：** [BootLoops](../../wiki/entities/bootloops.md)

## 核查依据与结构

在上述固定 commit 阅读 README.md、INSTALL.md 和 tools/README.md。49 个 tools 包涵盖精确计算、球算术、递推、积分与验证；tools 索引区分 selftest、partial、smoke、data-gated，不能把 skip/refused 当成完整验证成功。

| 目录 | 内容 |
|------|------|
| tools/ | 每包 GUIDE、入口与验收测试 |
| toolkit/ | 工具目录、配方、外部引擎来源 |
| upgrades/ | SOFIA.jl、Eichler.jl、Leviathan 等自有引擎 |
| ops/turnstile | 长作业资源准入；独立 selftest，不在工具总测试中 |

skills、jackandjill、kira、blade、amflow-cpp 为同组织独立仓库。逐问题计算代码在官网问题页；本次没有验证其下载可用性。

## 安装与测试入口（仅归档，未执行外部代码）

```bash
git clone https://github.com/BootLoops-ai/bootloops.git
cd bootloops
git checkout 66b680ce742e654cfe86da4f072a69061fe182b1
python3 -m venv .venv
source .venv/bin/activate
pip install mpmath sympy numpy python-flint pytest
python3 run_selftests.py --par 8
```

Python 3.12；部分包需要 Julia >=1.11 或额外依赖、外部引擎。README 要求先读 tools 索引和包 GUIDE，再提出计划供人批准。模型访问另行提供。

## 风险与归因

输入文件可能被解释为代码，应在可信输入或隔离环境中运行。数值证书不是安全沙箱；也不是临床、公共安全或监管用途认证。按实际包逐项审计许可，不用主仓 MIT 覆盖第三方文件。
