# Direct Perception Control Model（Symbiosis Robotics Blog）

> 来源归档（ingest · blog / 公司研究报告）

- **标题：** Direct Perception Control Model
- **类型：** blog
- **作者 / 组织：** Symbiosis Robotics Team
- **原始链接：** <https://symbiosis-robotics.com/research/dpc/en/>
- **规范引用：** <https://symbiosis-robotics.com/research/dpc>
- **发表日期：** August 2026
- **入库日期：** 2026-08-17
- **抓取方式：** 官方英文页直连（HTML 交叉核对；非 PDF）
- **配套项目页：** [`sources/sites/symbiosis-robotics-dpc.md`](../sites/symbiosis-robotics-dpc.md)
- **一句话说明：** 去掉分层全身系统的中间运动接口，让感知–控制在关节目标空间联合学习，并用 DriftDistill 扩大可执行分布。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-08-17） |
|----|-------------------------|
| 项目页 | <https://symbiosis-robotics.com/research/dpc/en/> |
| 代码 / 权重 / 数据 | **确认未开源**：页内无 GitHub / HF / 下载 |
| 论文 | **无 arXiv/PDF**；Citation 为公司博客 `@article{symbiosis2026directperceptioncontrol}` |
| 可信度边界 | 官方技术叙事 + 定性真机视频；**无**公开成功率表、消融数字或独立评测 |

## 核心摘录（归纳，非全文）

### 分层路径的三条瓶颈

多数全身系统仍是 System 1（任务理解 → 运动目标 \(Z_t\)）+ System 0（结合当前身体状态跟踪）。页内把这套栈的失败拆成三层，而不是「64 维太小」：

1. **运动接口不足。** \(Z_t\) 监督的是「身体应呈现何种运动」，不是「本控制步该发哪个关节目标」。任务相关控制信息一旦在运动学抽象里丢掉，下游 tracker 无法恢复。页内用 Pair Explorer 显示：latent 近邻在关节目标空间可以分得很开——接触、平衡、上下身协调发生在接口之后。
2. **分训合推。** 推理时看起来是一条链，训练目标并不连续：System 1 学接口、System 0 学跟踪/稳定；执行误差与视觉后果过不了冻结接口，无法共同优化 System 1。
3. **低层动作边界。** 最终动作必须落在冻结解码器对合法 SONIC latent 的像 \(M_h\) 内。若 System 0 主要吃 locomotion-centric 数据，新的 loco-manipulation 动作即使高层更强也到不了。

页内把对照解码器写死为 **冻结 SONIC**：\(L_{\mathrm{token}}=\delta z^\top I_{64}\delta z\)，\(L_{\mathrm{action}}\approx\delta z^\top J_h^\top J_h\delta z\)，\(J_h=\partial D_{\mathrm{frozen}}(z,h)/\partial z\in\mathbb{R}^{29\times 64}\)，故 \(\mathrm{rank}(J_h^\top J_h)\le 29<64\)。Direct-joint 把监督放到最终 29 维 PD 空间，并让未来视觉 token attend 被关节目标监督的 \(E_{A,t}\)，从而 \(\partial L_{\mathrm{total}}/\partial E_{A,t}\) 同时含动作损失与未来视觉损失。

**自由度与代价：** 同一 PD 接口下放大假设类，等于放弃解码器自带的平衡/平滑/全身协调先验，必须用数据与闭环训练重新学回来。限位、力矩饱和、安全裁剪仍在。

### Direct Perception Control

单一模型：视觉、语言、身体状态、动作历史、执行反馈 → **可执行关节与手部目标**。中间运动表示被移除。

- **Symbiotic Attention：** 感知表示与控制表示在共享动作目标下互相关注——任务理解受可执行性约束，动作生成持续抽语义上下文。
- **闭环节奏：** 视觉异步更新，本体反馈连续。
- **DriftDistill：** Student 先 Offline BC；Stage 2 把策略访问到的漂移状态与离线示范一起训；Frozen Teacher 给 \(a_t^*\)，最小化恢复损失 \(L_{\mathrm{rec}}(a_t,a_t^*)\)。循环：Visit → Correct → Absorb。

### 数据

异构人体/机器人语料不能直接混（本体、动作表示、控制接口不同）。全部转成 **G1 可执行、时间对齐关节轨迹**，共 **15,010 小时**：

| 来源 | 小时 |
|------|------|
| Human Ego | 6,781 |
| Armed Robot | 4,024 |
| Wheeled Humanoid | 3,660 |
| Bipedal Humanoid | 545 |

转换取各源最可靠运动信号：遥操作标准化坐标系/关节定义/控制率；头戴记录重建并重定向；egocentric 视频从手/腕/末端抬到全身。

### 演示（定性）

- 移动拾放：走近锥桶、蹲抓、转身走两步放下（抓取 + 负重平衡 + 转向运输）。
- 受限全身 loco-manipulation：紧空间里躯干与手臂重配置，保持支撑与物体接触。
- 手–眼–脚：右脚控油门、双手转向出弯，视觉时序 + 油门精度 + 驾驶语义。

页内**没有**公开成功率、对照表或消融。

## 对 wiki 的映射

- [paper-dpc](../../wiki/entities/paper-dpc.md) — 本篇主升格实体页
- [loco-manipulation](../../wiki/tasks/loco-manipulation.md) — 新增「去掉运动接口」技术路线
- [sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md) — 被本页当作冻结 System 0 的具体靶
- [vla-with-low-level-controller](../../wiki/queries/vla-with-low-level-controller.md) — 相对 VLA+WBC 异步的第四条架构读法
- [paper-motionwam-humanoid-loco-manipulation-wam](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) — 页内引用的分层 WAM 例
- [paper-omega-0](../../wiki/entities/paper-omega-0.md) — 页内引用的 SONIC latent WAM 例
- [gemini-robotics](../../wiki/entities/gemini-robotics.md) — 页内引用的分层全身 VLA 产品例
- [pi07-policy](../../wiki/methods/pi07-policy.md) — 页内引用的异构数据对照

## 可信度与使用边界

- **公司研究博客**，不是 peer-reviewed 论文；定量几乎全是演示叙事。
- **不要**写成可本地训练/复现的基线；截至入库日无代码、权重、数据集。
- 「去掉运动接口」≠ 卸掉 PD/安全层；页内明确保留 29 维 PD 目标与安全裁剪。
- 对 SONIC 的批评是**信息论/流形**层面的论证，未给出与公开 GEAR-SONIC 权重的对照实验。

## Citation

```bibtex
@article{symbiosis2026directperceptioncontrol,
  author  = {{Symbiosis Robotics Team}},
  title   = {Direct Perception Control Model},
  journal = {Symbiosis Robotics Blog},
  month   = {August},
  year    = {2026},
  url     = {https://symbiosis-robotics.com/research/dpc},
}
```
