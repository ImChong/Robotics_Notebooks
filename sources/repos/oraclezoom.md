# OracleZoom（dipta007 官方实现）

> 来源归档

- **标题：** OracleZoom — Official Code
- **类型：** repo
- **代码：** <https://github.com/dipta007/OracleZoom>
- **论文：** <https://arxiv.org/abs/2609.06490>
- **项目页：** <https://dipta007.github.io/OracleZoom/>
- **权重：** <https://huggingface.co/dipta007/OracleZoom>
- **Demo：** <https://huggingface.co/spaces/dipta007/OracleZoom>
- **入库日期：** 2026-09-28
- **一句话说明：** 递归 **4×** 链至 **256×** 的参考约束 SR：推理 `scripts/infer.py` → `opd_zoom.teacher.oracle_infer`；训练 `prepare_data.py` + `train_final.sh`；评测 `eval_final.sh`（与 CoZ 公平同 pipeline 对比）。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [机器人视觉感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md) | **高倍 zoom / 细节合成** 与 **观测一致性** 的选型语境（非机器人专论，但影响远距感知、质检、显微等下游） |
| [Vision Backbones](../../wiki/concepts/vision-backbones.md) | 冻结 VLM prompter + 扩散 SR 骨干 + LoRA 适配的 **组合式视觉栈** 案例 |

---

## 运行入口（README 摘要）

| 阶段 | 入口 |
|------|------|
| 安装 | `git clone --recursive` → `uv sync` → `uv run hf download dipta007/OracleZoom --local-dir weights/oraclezoom` |
| 推理 | `uv run scripts/infer.py --input my_images/ --output out/`（512×512 中心裁剪，4 步 4× → scale1–4 文件夹） |
| 数据 | `uv run scripts/prepare_data.py --tier 1k --out data/train_1k --eval_layout data/eval_val` |
| 训练 | `scripts/train_final.sh data/train_1k ... out/run1`（约 9300 step 早停） |
| 评测 | `scripts/eval_final.sh data/eval_val renders/ours results/ours` |

**注意：** 发布权重为 **merged transformer**，推理用 `--full_transformer`，勿传 `--pld_lora`。

---

## 对 wiki 的映射

- 主实体页：**`wiki/entities/paper-oraclezoom.md`**
- 论文摘录：**`sources/papers/oraclezoom_arxiv_2609_06490.md`**
- 项目页：**`sources/sites/oraclezoom-project.md`**
