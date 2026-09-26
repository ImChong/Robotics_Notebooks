# motion-jepa

> 来源归档

- **标题：** motion-jepa
- **类型：** repo
- **链接：** <https://github.com/mkarmann/motion-jepa>
- **论文：** <https://arxiv.org/abs/2609.23881>
- **项目页：** <https://mkarmann.github.io/motion-jepa-project-page/>
- **许可：** 仓库根目录 **未声明 LICENSE**（README 标注论文 CC BY 4.0）；上游 LeWM 组件见 [le-wm](../../sources/repos/le-wm.md)
- **入库日期：** 2026-09-26
- **一句话说明：** MotionJEPA 官方实现：三合成游戏 JEPA 训练、离线 state probing 与 LeWM 族基线；CEM 静态干扰规划在 `planning/` git 子模块。
- **沉淀到 wiki：** [`wiki/entities/paper-motionjepa.md`](../../wiki/entities/paper-motionjepa.md)

---

## 仓库入口（README，2026-09-26）

| 组件 | 说明 |
|------|------|
| 环境 | Python **3.11**；[uv](https://docs.astral.sh/uv/) |
| 克隆 | `git clone --recurse-submodules https://github.com/mkarmann/motion-jepa.git` |
| 安装 | `uv sync` |
| 单环境训练 | `bash run.sh pong motionjepa ./runs/pong_motionjepa` |
| 三环境 × 三 seed | `bash run_all_environments_three_seeds.sh motionjepa` |
| 分步 | `train.py` → `train_probes.py` → `evaluate.py` |
| 数据集类型 | `pong`, `dino`, `golf` |
| 模型类型 | `motionjepa`, `lewm`, `smwm`, `lewm-sigreg-time`, `lewm-tgt-detach`, `lewm-sigreg-flattened` |
| 超参 YAML | `configs/`（含 optimized 配方） |
| 规划 | `planning/` **子模块**（README：downstream planning results are in the planning subrepo） |

---

## 开源边界

| 已发布 | 备注 |
|--------|------|
| 主仓训练 / probing / 评测脚本 | 无根目录 LICENSE 文件 |
| LeWM 架构与 SIGReg 致谢链 | 来自 `lucas-maes/le-wm` |
| 规划 CEM + 四控制任务 distractor 实验 | 需初始化子模块 |
