# Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization

> 来源归档（ingest）

- **标题：** Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization
- **短名：** Zero-WAM
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.26103>
- **PDF：** <https://arxiv.org/pdf/2608.26103>
- **项目页：** <https://robbyant-research.github.io/Zero-WAM/>
- **代码：** <https://github.com/robbyant-research/Zero-WAM>
- **作者：** Jiaming Zhou, Qihang Zhang, Gangwei Xu, Cunxin Fan, Yujie Zhao, Ruilin Wang, Yiming Luo, Shuai Yang, Xing Zhu, Yujun Shen, Junwei Liang, Yinghao Xu
- **版本：** arXiv:2608.26103v2 [cs.RO] 27 Aug 2026
- **入库日期：** 2026-08-28
- **再核日期：** 2026-11-19
- **核对范围：** 通读 `arXiv:2608.26103v2` 全文（1687 行，含 §1–§6、HumanGen 管线、MoT 架构与 ΔH RoPE 偏移、IFP/MCP 目标、RoboTwin 2.0/真机 Franka 与消融实验），并核对项目页与开源仓库源码
- **通读：** 已完成
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 把人类视频当 in-context 任务规格，因果视频-动作模型零样本执行未见操作任务。

## 开源状态（步骤 2.5）

- **已开源（代码 + 预训练/RoboTwin 后训练权重 + 部分 HumanGen 潜变量数据集，2026-11-19 再核）**：
  - 官方仓库 [`robbyant-research/Zero-WAM`](https://github.com/robbyant-research/Zero-WAM)（Apache-2.0）已公开完整训练、服务化与 RoboTwin 评测源码（`wan_va/`、`script/train.sh`、`evaluation/robotwin/`）。
  - Hugging Face 已发布 `robbyant-research/zero-wam-pretrain`、`robbyant-research/zero-wam-posttrain-robotwin` 权重与 `datasets/robbyant-research/HumanGen`（含 RoboTwin ICL 与 5 份外部预训练潜变量数据；内部/真机子集及 HumanGen 合成生成脚本未包含在公开仓内）。
  - 仓库 `INSTALL.md` 与论文 Footnote 1 明确：`move_stapler_pad`（取消垫内偏航角限制）与 `stamp_seal`（目标误差容限从 `1 cm` 放宽至 `3 cm`）两项评测文件存在协议修订，与所有基线公平对齐。

## 核心摘录（面向 wiki 编译）

### 摘录 1：人类视频作任务规格

- 零样本跨任务泛化被改写成 in-context 任务指定；语言不足以描述物体交互过程，人类视频提供任务演化线索。
- HumanGen：自动把任务采样的机器人轨迹转成语义对齐人类视频，**7.42 万**人机配对、覆盖 **8600** 个任务。
- IFP（in-context future chunk prediction）抑制从已见任务抄近道，迫使模型从视频提示抽取任务信息。
- 单一策略同时支持语言指令与人类视频提示。

**对 wiki 的映射：** [paper-zero-wam](../../wiki/entities/paper-zero-wam.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[机器人 In-Context Learning](../../wiki/concepts/robot-in-context-learning.md)

### 摘录 2：评测

- RoboTwin 2.0 七个未见任务平均成功率 **46.95%**（摘要四舍五入为 47.0%；相对最强视频动作基线 LingBot-VA **17.45%**，+29.5 pp）。
- 真机双臂 Franka、无需对应机器人数据或参数更新：物体入容器 **53.3%**、三物体顺序操作 **33.3%**、双桌腿插入 **16.7%**（对照 LingBot-VA 分别为 43.3 / 10.0 / 0.0）。

**对 wiki 的映射：** [manipulation](../../wiki/tasks/manipulation.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-zero-wam.md`](../../wiki/entities/paper-zero-wam.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
