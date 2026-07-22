# DA-Nav: Direction-Aware City-Scale Vision-Language Navigation（arXiv:2607.11638）

> 来源归档（ingest）

- **标题：** DA-Nav: Direction-Aware City-Scale Vision-Language Navigation
- **类型：** paper / VLN / outdoor navigation / VLM / sim2real / recovery
- **来源：** arXiv abs / PDF / HTML（v2，2026-07-14；v1 2026-07-13）
- **原始链接：**
  - <https://arxiv.org/abs/2607.11638>
  - PDF：<https://arxiv.org/pdf/2607.11638>
  - HTML：<https://arxiv.org/html/2607.11638v2>
- **作者：** Ye Yuan\*, Kehan Chen\*, Xinqiang Yu\*, Wentao Xu, Heng Wang, Libo Huang, Chuanguang Yang, Yan Huang, Jiawei He†, Zhulin An†（\*同等贡献；†通讯作者）
- **机构：** 上海科技大学（ShanghaiTech）；中国科学院计算技术研究所（ICT, CAS）；中国科学院自动化研究所 NLPR（CASIA）；中国科学院大学（UCAS）；XYZ Embodied AI
- **入库日期：** 2026-07-22
- **一句话说明：** 用商业导航工具的**粗粒度方向指令**做城市尺度户外 VLN；将导航改写为 **egocentric 图像平面离散 spatial grounding**，并以 **CoT（偏离评估→动作→网格轨迹）** 支撑闭环恢复；配套 **ReDA** 数据集（含 recovery）；CARLA SoTA，并 **零样本** 迁移 Unitree Go2 / 乐聚 Kuavo-V，公里级户外闭环。

## 开源状态（核查 2026-07-22）

- **未开源 / 无项目页：** arXiv abs、HTML、PDF 均未列出 GitHub、Hugging Face、项目页或「code will be released」外链；公开 Web 检索未发现官方同名可运行仓。
- **复现边界：** 方法依赖 **Qwen2.5-VL-7B-Instruct + LoRA**、自建 **ReDA（CARLA）** 与远端 GPU 推理；无公开权重/数据时无法按官方入口复现。
- **互指：** 升格实体页 [`wiki/entities/paper-da-nav.md`](../../wiki/entities/paper-da-nav.md)

## 摘要级要点

- **范式：** Direction-Aware VLN——输入 egocentric RGB 历史（k=4）+ 离散方向指令 \(I_t\in\{\texttt{FORWARD},\texttt{TURN\_LEFT},\texttt{TURN\_RIGHT},\texttt{STOP}\}\)（来自 Google Maps / 高德等解析），输出可执行局部轨迹。
- **关键改写：** 不做连续 3D waypoint 回归，而在 egocentric 图像平面构造可通行网格 \(G=\{(r,c)\mid r\in[13,23],\,c\in[0,28]\}\)，预测长度 \(L=6\)（约未来 3 s @2 Hz）的离散网格序列 \(\mathbf{P}_t\)。
- **CoT 决策序列：** \(Y_t=(s_t,c_t,\mathbf{P}_t)\)——\(s_t\in\{\texttt{Yes},\texttt{No}\}\) 是否偏离；\(c_t\) 含 FORWARD / TURN\_\* / CORRECT\_\* / STOP；再预测网格轨迹。
- **ReDA：** CARLA 自动管线；Stable / Drifting / Recovering FSM；注入转向扰动；丢弃 Drifting 帧；约 **286k** 样本（158k expert + 128k recovery）；2102 轨迹 / 126 场景。
- **策略：** 冻结视觉编码器；**LoRA** 微调 Qwen2.5-VL-7B；图像网格→机体系（深度或 IPM 回退）；**furthest-point** 控制接口降 VLM 延迟抖动。
- **仿真（239 条闭环）：** SR **59.00%**、SPL **58.66**、CSR **98.15%**；相对 CityWalker / ViNT / NaVid / NaVILA / 零样本 Qwen2.5-VL 全面更强恢复能力；摘要另报未见城镇 SR **56.16%**。
- **消融：** 去掉 recovery 数据 → SR 29.71%、CSR 15.46%；去掉 CoT → SR 38.91%、DF 升至 4.30。
- **真机：** RealSense D455 + 手机侧商业导航指令解析；远端 RTX 4090；开环原语平均 SR **83.3%**；城市/公园闭环整体 SR **46.7%**；人形 **Kuavo-V 零样本 1.2 km** 户外导航。

## 核心论文摘录（MVP）

### 1) 任务：Direction-Aware VLN（商业导航指令接地）

- **链接：** <https://arxiv.org/abs/2607.11638> §I–III-A
- **摘录要点：** 城市尺度地图/SLAM 成本高；细粒度语言/地标监督难扩展；商业导航工具已提供全局规划与实时方向提示，但「前方 50 米右转」对人友好、对机器人不可直接执行。本文将其形式化为方向感知 VLN，挑战是 **稀疏方向→局部可执行行为** 与 **长程误差累积后的恢复**。
- **对 wiki 的映射：**
  - [DA-Nav](../../wiki/entities/paper-da-nav.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)

### 2) 方法：图像平面离散 grounding + CoT 恢复

- **链接：** §III-C、§IV
- **摘录要点：** 有效网格避开天空与远景；结构化 prompt 强制「先评估偏离、再选动作、最后选网格」；相对连续回归更贴合 VLM 的 2D 视觉推理；机体系投影用深度优先、不可靠时 IPM；控制取预测视界最远点并按转向幅值自适应降速。
- **对 wiki 的映射：**
  - [DA-Nav](../../wiki/entities/paper-da-nav.md)
  - [VLA](../../wiki/methods/vla.md)
  - [VLN 四范式复现路径](../../wiki/overview/vln-open-source-repro-paradigms.md) — 对照 Uni-NaVid / NoMaD 等动作表示选择

### 3) 数据：ReDA（方向指令 + recovery）

- **链接：** §III-B–D；Table I
- **摘录要点：** 相对 R2R / Touchdown / CityWalker / NaVILA 等「细粒度描述 / 专家轨迹」数据集，ReDA 强调 **direction-based 指令 + 2D image grid 动作空间 + expert+recovery**；FSM 在横向误差 \(e_y\geq 0.35\) m 进入 Recovering，自适应 look-ahead 校正。
- **对 wiki 的映射：**
  - [DA-Nav](../../wiki/entities/paper-da-nav.md)
  - [行为克隆](../../wiki/methods/behavior-cloning.md) — 纯专家 BC 在 OOD 偏离下 CSR 崩溃的反例

### 4) 评测：恢复指标 DF / CSR + 跨具身零样本

- **链接：** §V；Table II–V；Fig. 8
- **摘录要点：** 常规 SR/RC/SPL 不足刻画闭环纠偏；定义 DF（每百米偏离事件）与 CSR（成功纠回率）；NaVid/NaVILA 虽 RC 不低但 CSR≈5%；真机相对 CityWalker/ViNT 闭环 SR 显著更高，并展示四足→人形跨具身。
- **对 wiki 的映射：**
  - [DA-Nav](../../wiki/entities/paper-da-nav.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [NaVILA（笔记实体）](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-da-nav.md`](../../wiki/entities/paper-da-nav.md) — 主实体页
- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) — 城市尺度 / 方向指令 VLN 分支
- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md) — CARLA→足式/人形零样本户外导航
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — VLM/VLA 导航动作表示对照
- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md) — 与四范式开源栈对照（本文暂未开源）
- [`wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md`](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md) — 足式导航 VLA 基线对照

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.11638>
