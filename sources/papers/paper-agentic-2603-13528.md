# Learning Actionable Manipulation Recovery via Counterfactual Failure Synthesis

- **作者：** Dayou Li, Jiuzhou Lei, Hao Wang, Lulin Liu, Yunhao Yang, Zihan Wang, Bangya Liu, Minghui Zheng, Zhiwen Fan
- **机构：** Texas A&M University, University of Minnesota, University of Texas at Austin, Abaka AI, University of Wisconsin-Madison
- **版本 / 载体：** arXiv:2603.13528v1 [cs.RO]（2026-03-13 提交，HTML 全文版）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2603.13528) / [arXiv HTML 全文](https://arxiv.org/html/2603.13528)
- **项目页 / 代码：** 未确认开源代码（论文主页附带地址 [dream2fix.github.io](https://dream2fix.github.io/)；预抓取 `project_page_status: 0`，论文正文与检索均未提供可验证的官方 GitHub 仓库或可运行源码入口）
- **核对范围：** 通读 arXiv HTML 全文全部章节（Abstract、I Introduction、II Related Works、III Method、IV Experiment、V Conclusions 与 References），核对表 I–IV 全部数值与公式 (1)–(13)。
- **通读：** 已完成

## 原文核对笔记

- **核心定位与问题：** 针对 VLA/VLM 在真实操作任务中缺乏执行失败后自主恢复能力、真机收集失败数据成本高且危险、仿真扰动存在明显 Sim-to-Real 鸿沟以及现有视觉分析器仅输出粗粒度二值判定而非可执行轨迹修正的问题，提出 **Dream2Fix** 框架。
- **反事实失败合成（III-B-1）：** 基于动作条件生成式世界模型 **Ctrl-World**（arXiv:2510.10125），以夹爪开合切换时刻为关键帧（keyframe），在其局部时间窗口内对成功真机演示注入四类可控动作扰动：`delay_close`（延迟闭合）、`weak_close`（闭合幅度不足）、`force_open`（关键帧夹爪动作反转保持打开）与 `translation`（接近与接触阶段末端 $x$-$y$ 状态加入高斯噪声）。
- **四维结构化验证器（III-B-2）：** 为滤除世界模型幻觉与物理不可行样本，串联四类校验器：
  1. **VLM Verifier（Qwen3-VL-8B）**：对照任务指令、成功参考视频与生成失败视频，核查任务失败有效性及视觉质量；
  2. **IDM Verifier（ResNet-50 + MLP）**：回归相邻帧末端状态变化 $\widehat{\Delta s}_{t,d}$ 并与条件动作状态差 $\Delta s_{t,d}$ 比较，要求 $e_{t,d} \le \tau$；
  3. **Joint Pose Verifier（ResNet-50）**：估计 7 自由度关节角 $q_t$ 及差分速度 $\dot{q}_t$、加速度 $\ddot{q}_t$，校验关节限位与阈值 $(\tau_v, \tau_a)$；
  4. **Point-Tracking Verifier（CoTracker3）**：在初始帧 $10 \times 10$ 网格追踪像素轨迹，计算 $S_{\text{smooth}}$、$S_{\text{vis}}$、$S_{\text{topo}}$、$S_{\text{global}}$ 加权总分 $S_{\text{pt}}$。
  全流程筛除 41.9% 合成样本、保留 58.1%，构建超过 120k（120K）条配对失败-修正样本。
- **确定性修正标注与闭环策略微调（III-B-3 & III-C）：** 直接依据注入的扰动参数生成固定键值格式（`RESULT`、`TYPE`、`STAGE`、`FIX_DIR_*`、`FIX_N_*` 或夹爪时序锚点）加自然语言摘要的修正标签；基于 `Qwen3-VL-8B-Instruct` 冻结视觉编码器并微调多模态投影器与语言模型（得到 **Dream2Fix-VLM**）；在 Franka Research 3 + Franka Hand + RealSense D435i 平台通过固定动作预算事件触发机制与确定性控制原语映射实现闭环恢复（并接入 OpenVLA 实机部署评测）。
- **实证关键数据（IV）：** 表 II 对比真实成功演示与保留合成样本的校验指标分布接近；表 III 中 Dream2Fix-VLM 在合成测试集 Correction Accuracy（`cap=3`）由 GPT-4o 的 0.197 提升至 0.813，在 100 条 10 任务真机基准上零样本达到 0.472（Gemini-3-Flash 为 0.374）；表 IV 真机闭环恢复率（每组 50 次）在 Real-World Benchmark 上为 46%（vs Gemini3-Flash 36%），在 OpenVLA 部署失败恢复上为 40%（vs Gemini3-Flash 30%）。
- 对 wiki 的映射：[Learning Actionable Manipulation Recovery via Counterfactual Failure Synthesis](../../wiki/entities/paper-agentic-2603-13528.md)
