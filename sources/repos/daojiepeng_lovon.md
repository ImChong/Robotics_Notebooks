# DaojiePENG/LOVON 官方代码仓库核对归档

> 来源归档（GitHub 源码实体核对）

- **仓库：** <https://github.com/DaojiePENG/LOVON>
- **类型：** repo
- **许可证：** MIT（根目录 `LICENSE`）
- **核对范围：** 核对 `README.md`、`LICENSE`、`deploy/lovon_deploy.py`、`models/api_object_extraction.py` 与 `models/api_language2mostion.py`。
- **通读：** 已完成
- **入库日期：** 2026-11-14

## 源码架构与差异核对

1. **三线程实时部署架构（`deploy/lovon_deploy.py`）**：面向 Unitree 足式平台搭建 Tkinter GUI 交互入口，由 `ImageGetterThread`（含 `AdaptiveBlurThreshold` 自适应模糊帧过滤）、`YoloProcessingThread`（开放词汇/目标框检测）与 `MotionControlThread` 协同运行，最终调用 `unitree_sdk2py` 的 `SportClient.Move` 下发速度指令。
2. **模型推理接口（`models/`）**：`api_object_extraction.py` 封装 `SequenceToSequenceClassAPI`（IOE 指令目标提取）；`api_language2mostion.py`（注意仓库实际文件名含 `mostion` 拼写笔误，而 README 注释写作 `api_language2motion.py`）封装 `MotionPredictor`（L2MM）。
3. **LLM 闭环部署尚未接入公开脚本**：`README.md` 的 TODO 清单仍显示 `- [ ] Deployment with LLM` 未勾选；当前公开 `lovon_deploy.py` 仅支持人工/单任务指令切换，论文中的在线 DeepSeek-R1 多子目标动态重规划闭环尚未随该分支源码发布。

## 对 wiki 的映射

- [paper-notebook-lovon-legged-open-vocabulary-object-navigator](../../wiki/entities/paper-notebook-lovon-legged-open-vocabulary-object-navigator.md)
