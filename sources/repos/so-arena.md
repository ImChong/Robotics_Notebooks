# SO-Arena 官方代码（holdrain/SO-Arena）

- **仓库：** <https://github.com/holdrain/SO-Arena>
- **论文：** <https://arxiv.org/abs/2611.15777v1>
- **官网：** <https://lichenlillc.github.io/SO-Arena/>
- **数据集：** <https://huggingface.co/datasets/SO-Arena/SO-Arena>
- **论文归档：** [arXiv:2611.15777](../papers/so_arena_arxiv_2611_15777.md)
- **官网归档：** [官方主页](../sites/so-arena.md)
- **wiki 映射：** [SO-Arena](../../wiki/entities/paper-so-arena.md)
- **许可：** GitHub 代码 MIT；自有 annotations CC BY 4.0；各 benchmark 样本保留第三方许可证。
- **核对日期：** 2026-11-19
- **核对版本：** GitHub 公开 main；锁定 data-lock.json 的数据修订及哈希，当前代码未另钉 SHA。
- **核对范围：** README、根 jev、so_arena.py、evaluator/README、evaluator/MODEL_ADAPTER.md、evaluator/jevbench.py、reliability/README、build_variants.py、run_variants.py、score.py。
- **通读：** 已完成

## 已发布组件与真实运行入口

| 文件 | 核对到的职责 |
|---|---|
| jev（根目录 shell） | 转发至 Python 3.12 so_arena.py |
| so_arena.py | fetch-data 从 HF 锁定修订下载并校验 SHA-256；prepare 将代码与数据拼装为 runtime；将 run/score 等转发 runtime/jev |
| evaluator/jevbench.py | setup、verify、doctor、self-test、list、run、score；执行 static/web/tools 评分和汇总 |
| evaluator/MODEL_ADAPTER.md | adapter measure/predict 或 HTTP /measure、/predict；模型只收到公开 state/questions，不收到私有答案 |
| reliability/build_variants.py | 939 个选项顺序测试（原顺序+12 种排列）；1,052 个缺答案测试候选，其中 597 个客观域条目用于 N |
| reliability/run_variants.py | 将变体交给模型 adapter，记录每次返回与错误 |
| reliability/score.py | 将概率映射回原选项，计算 S、N、R；并列最大概率分摊，失败不无声删除 |
| data-lock.json / code-files.json | 锁定数据和评估器的版本完整性 |
| latency/ | 1,109 个计时任务及单张 H100 的测量协议 |

## 官方命令（静态核对，未在本机执行）

~~~bash
git clone https://github.com/holdrain/SO-Arena.git
cd SO-Arena
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install huggingface_hub
./jev fetch-data --output "$PWD/../so-arena-data"
export SO_ARENA_DATA="$PWD/../so-arena-data"
export SO_ARENA_WORKSPACE="$PWD/../so-arena-runtime"
./jev verify && ./jev setup && ./jev doctor --output "$PWD/doctor.json"
export JEV_ENDPOINT=http://localhost:8000
./jev run --adapter adapters:http --model your-model --out "$PWD/runs/your-model"
./jev score --run "$PWD/runs/your-model" --output "$PWD/runs/your-model/report.json"
~~~

独立可靠性评估：

~~~bash
python3 reliability/build_variants.py --data "$SO_ARENA_DATA" --out variants --check reliability/variants.sha256.json
python3 reliability/run_variants.py --variants variants --adapter adapters:http --model your-model --out preds/your-model
python3 reliability/score.py --variants variants --predictions preds/your-model --csv scores.csv --out scores.json
~~~

## 口径核对与边界

- 默认 leaderboard 选 1,232 个任务；每域先平均再 11 域平均，不按题数加权。
- S 计算答案内容的重排一致性；N 在 597 个客观域任务上按 AUC 得分；R 是二者几何平均。
- 原始答案、checker state、reference route 不发送给模型；异常运行和拒绝回答有明确记录。
- 数据含受保护项目及许可证限制；官方明确禁止测试集进入模型训练语料。
- 尚未运行 fetch-data、setup、真实模型评估或 GPU 计时；本文只说明公开代码可确认的执行路径。
