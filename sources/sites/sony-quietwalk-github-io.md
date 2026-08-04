# QuietWalk（Sony aibo）官方项目页

> 来源归档（项目页核查）

- **标题：** Learning Quiet Walking for a Small Home Robot（ICRA 2025 QuietWalk）
- **类型：** site / project-page
- **官方入口：** <https://sony.github.io/QuietWalk/>
- **GitHub Pages 源仓：** <https://github.com/sony/QuietWalk>（仅学术项目页模板 + 静态资源，**无可运行训练/部署代码**）
- **论文：** <https://arxiv.org/abs/2502.10983>
- **机构：** 苏黎世联邦理工机器人系统实验室（ETH Zürich RSL）；索尼集团（Sony Group Corporation）；新加坡国立大学（NUS）等
- **入库日期：** 2026-08-02
- **一句话说明：** 在 Sony **aibo** 上用 sim-to-real RL 惩罚 **足端接触速度**，配合 **可变 PD gain**、**足底开关接触传感** 与 **两阶段课程**，实现比索尼商用控制器更安静的家庭四足行走。
- **代码：** 项目页与 `sony/QuietWalk` 仓**未提供**训练脚本、权重或部署入口。
- **开源状态（2026-08-02 核查）：** **确认未开源可运行实现**；GitHub 仓仅为项目页（`index.html` / `static/` / 模板 README），无 Isaac Gym 训练或真机推理代码。

## 页面公开信息

- 视频对比：提出的 RL 策略相对 RL baseline 与 Sony normal / quiet 商用控制器更安静。
- 鲁棒性：更安静策略在陡坡（约 7°）上弱于更响亮的 baseline；可用 Domain Randomization 参数调节安静度–鲁棒性权衡。
- 方法三点：足底开关接触传感、策略输出关节目标 + PD gain scale、先学走再加重噪声惩罚的课程。

## 对 wiki 的映射

- 论文归档：[learning_quiet_walking_aibo_arxiv_2502_10983.md](../papers/learning_quiet_walking_aibo_arxiv_2502_10983.md)
- 实体页：[paper-learning-quiet-walking-aibo.md](../../wiki/entities/paper-learning-quiet-walking-aibo.md)
