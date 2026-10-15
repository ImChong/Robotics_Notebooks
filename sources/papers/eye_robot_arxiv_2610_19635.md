# Eye, Robot: Learning Expressive Gaze Dynamics via a VLM-Guided Transformer-VAE（arXiv:2610.19635）

> 来源归档（ingest）

- **标题：** Eye, Robot: Learning Expressive Gaze Dynamics via a VLM-Guided Transformer-VAE
- **arXiv：** <https://arxiv.org/abs/2610.19635>（PDF：<https://arxiv.org/pdf/2610.19635>；HTML：<https://arxiv.org/html/2610.19635>）
- **提交日期：** 2026-10-14（v1，cs.RO；交叉列 cs.AI；CC BY 4.0）
- **代码（文章所附）：** <https://anonymous.4open.science/r/eye_robot_pub-CF64/>（匿名仓库链接）
- **作者：** Yuhang Hu, Junlin Lyu, Yuan Fang, Xiaoying Shen, Lechen Zhang, Jiong Lin, Judah Goldfeder, Steven Tian, Hod Lipson
- **机构：** Columbia University，Department of Mechanical Engineering，Creative Machines Lab（全部作者同一单位）
- **入库日期：** 2026-10-15
- **一句话说明：** 用异步 VLM 选择场景指代目标、感知模块维持注视，再由 Transformer-VAE（Realistic Gaze Transformer，RGT）根据目标生成头颈与眼部协同运动，并用辅助眨眼预测驱动眼睑模板；在人形机器人上做集成演示。

## 开源状态（步骤 2.5，2026-10-15）

- **论文页核查：** arXiv 页面只提供通用链接（HTML、PDF、TeX 源码、音频摘要）与平台徽章，未列出本论文的项目页或代码仓库。
- **文章所附链接：** 匿名仓库 `anonymous.4open.science/r/eye_robot_pub-CF64/`。本次抓取仅返回 "Anonymous Github" 标题，README、文件列表、运行说明与权重/数据情况均不可读。
- **结论：** **未核实**（匿名链接内容不可读；论文页未列项目代码）。匿名链接不能视为正式开源，待作者公开正式仓库后再更新。

## 摘录 1：问题与主张（Abstract / §1）

- **痛点：** 注视通过眼、头、眼睑耦合运动来引导注意力；固定跟踪规则无法体现人类注视的时间结构。
- **主张：** 把问题拆为三部分：(i) 异步 VLM 选择场景指代；(ii) 反应式感知维持注视；(iii) RGT（Transformer-VAE）生成目标条件化的运动。

## 摘录 2：方法（§3）

- **VLM 引导的注视目标选择（§3.1）：** 用现成的视觉语言模型做开放词汇目标选择（论文页未给出模型名以外的细节，本页不展开）。
- **数据（§3.2）：** 无标记人体录制，重定向到机器人 11 自由度关节空间。
- **眼球轨迹生成（§3.3）：** 四维双眼轨迹，驱动颈部与眼部协同运动。
- **RGT（§3.4）：** 先用最近的全头运动历史做自监督预训练，再做自回归微调，使模型学习自身预测的 rollout 误差。
- **训练流程（§3.5）：** 上述两阶段。辅助眨眼预测器触发眼睑模板。

## 摘录 3：实验（§4，转述）

- 两秒基准：所有控制器接收相同的真值目标；RGT-Base 报告的七关节总体 MSE 最低。**页面摘要未给出具体数值，本页不写数字。**
- 真机演示：集成的目标选择与注视控制流程。
- 论文章节含定量结果、不同受试者适应性与消融（§4.1–4.3）；本次未逐项核实数值。
- 自然度感知与开放词汇目标选择准确率：论文自述尚需受控评测。

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-eye-robot-gaze.md`](../../wiki/entities/paper-eye-robot-gaze.md)。
- 与 [Eye Robot 2 主动注视操作（arXiv:2610.03710）](../../wiki/entities/paper-eyerobot-2-active-gaze-manipulation.md) 名称相近但不是同一论文，页面中需明确区分。
