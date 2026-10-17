# Intern-Decision（InternLM/Intern-Decision）

> 来源归档（repo；核对日期：2026-10-17）

- **代码：** <https://github.com/InternLM/Intern-Decision>
- **论文：** <https://arxiv.org/abs/2610.22131>
- **项目页：** <https://github.com/InternLM/Intern-Decision>
- **在线演示：** <https://huggingface.co/spaces/internlm/intern-decision>
- **模型权重：** <https://huggingface.co/collections/internlm/intern-decision>
- **代码许可：** Apache-2.0（仓库 LICENSE）；模型权重在独立模型卡发布，使用前应核对对应 checkpoint 的许可条款。
- **开源状态：** 已开放训练代码、Hugging Face 与 XTuner 两种推理后端、基准评测、温度校准、浏览器演示及合成示例。GitHub 仓库不含模型权重、训练数据、内部校准/验证记录、图片及数据准备流水线；用户需自行准备训练数据与基础 checkpoint。仓库附带七项准确率测试集与 96 条分布校准 pilot。
- **沉淀到 wiki：** [paper-intern-decision.md](../../wiki/entities/paper-intern-decision.md)

## 仓库入口核查

- `src/`、`configs/`、`scripts/`：请求推理、训练、评测与 checkpoint 验证入口。
- `runtime/xtuner/`：XTuner 推理兼容运行时；另有 Hugging Face 原生推理后端。
- `benchmarks/`、`docs/EVALUATION.md`：七项准确率套件、校准 pilot 与复现协议。
- `docs/DATA.md`：用户自备 JSONL 的运行时数据接口；图片通过 `MEDIA_ROOT` 定位。
- 训练配置支持 Qwen3.5 dense 0.8B、2B、4B 和 9B 结构；论文与公开权重集合聚焦 0.8B、2B、4B。

## 复现边界

推理需另行获取匹配的模型 checkpoint 与兼容 CUDA 环境。XTuner 训练还需安装固定版本 XTuner、Torch/CUDA 与 flash-attn；训练数据及准备过程未随仓库发布。README 的延迟结果是在单张 RTX 4090、BF16、SDPA 下，以 289 输入 token、三字段单次前向请求测得，不应泛化为所有硬件或请求长度的吞吐承诺。

## 互链

- [项目页与开源范围核查](../sites/intern-decision.md)
- [论文来源](../papers/intern_decision_arxiv_2610_22131.md)
- [Wiki 实体](../../wiki/entities/paper-intern-decision.md)
