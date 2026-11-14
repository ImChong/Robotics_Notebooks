# Real-Time Execution of Action Chunking Flow Policies（arXiv:2506.07339）

> 来源归档（ingest）

- **标题：** Real-Time Execution of Action Chunking Flow Policies
- **短名：** Real-Time Chunking / RTC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2506.07339>
- **项目页：** <https://www.pi.website/research/real_time_chunking>
- **代码（仿真）：** <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- **训练期后续：** <https://arxiv.org/abs/2512.05964> — [`training_time_rtc_arxiv_2512_05964.md`](training_time_rtc_arxiv_2512_05964.md)
- **LeRobot RTC 文档：** <https://huggingface.co/docs/lerobot/rtc> — [`../sites/lerobot-rtc-docs.md`](../sites/lerobot-rtc-docs.md)
- **社区 openpi 实现：** <https://github.com/EII-Tokyo/openpi-rtc> — [`../repos/openpi-rtc.md`](../repos/openpi-rtc.md)
- **机构：** 物理智能（Physical Intelligence）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-11-07（首次入库 2026-09-28）
- **版本：** arXiv:2506.07339v2（2025-12-05）；NeurIPS 2025
- **核对范围：** 2026-11-07 下载 arXiv v2 PDF（25 页）并通读：摘要、§1 引言、§2 预备与动机（Fig. 2–3）、§3 推理期 inpainting 与软掩码（式 1–5、Algorithm 1）、§4 实验（Kinetix 12 任务、真机 6 任务、基线）、§5 相关工作、§6 讨论与局限，以及附录 A.1–A.7（β 截断消融、延迟分解 Table 1–3、软掩码衰减与 Diffuser 对比 Fig. 8、超参 Table 4、算力）；参考文献仅浏览；真机视频和项目页博客未重新核对，仓库代码未运行。
- **通读：** 已完成
- **一句话说明：** 推理期把下一段动作 chunk 当成补全问题，让 flow / diffusion VLA 在执行旧 chunk 时生成与之衔接的新 chunk。

## 开源状态（步骤 2.5，2026-09-28）

- **部分开源**：Kinetix 仿真实验仓可跑专家数据、flow 模仿与延迟扫描，并含 training-time RTC 的微调步骤。权重与数据在 `gs://rtc-assets/`（`expert/` 约 60GiB）。
- 真机 π₀ / π₀.₅ 上的 inpainting 推理**未**作为 openpi 的独立训练入口发布。博客写明 π₀、π₀-FAST、π₀.₅ 的定量结果当时仍是同步执行。

## 核心摘录（面向 wiki 编译）

- π 系 chunk 为 50 步、约 1 秒。同步执行会在 chunk 之间停顿；停顿不在训练分布里。时间集成（ACT 的 temporal ensembling）在高延迟下会给出无效动作。
- RTC 在推理时冻结已经来不及改的前几步（对齐上一 chunk），对其余重叠步部分约束，再补全剩余动作。不改训练即可套到 flow / diffusion VLA。
- 博客示例延迟：移动操作远程推理合计约 139 ms（模型 97 ms），固定臂约 108 ms。注入 +200 ms 后 RTC 吞吐基本不变，同步推理与 temporal ensembling 明显下降。
- **对 wiki 的映射：** [paper-real-time-chunking](../../wiki/entities/paper-real-time-chunking.md)、[paper-training-time-real-time-chunking](../../wiki/entities/paper-training-time-real-time-chunking.md)；仓库 [real-time-chunking-kinetix](../repos/real-time-chunking-kinetix.md)、[openpi-rtc](../repos/openpi-rtc.md)

## 通读核对结论（2026-11-07，对照 arXiv v2 全文）

- **记号：** 预测视野 $H$，执行视野 $s\le H$（常 $s\approx H/2$），控制周期 $\Delta t$，推理耗时 $\delta$，**推理延迟** $d=\lfloor\delta/\Delta t\rfloor$。同步推理在每个执行视野结束后停下等待，$\delta>\Delta t$ 时出现停顿，改变机器人动力学并造成训练/部署分布偏移；朴素异步（在 $s-d$ 时启动、到达即切换）在 chunk 边界可能产生跳变与超高加速度（Fig. 2）。
- **约束：** 异步需 $d\le H-s$；论文设 $d\le s\le H-d$。
- **方法：** 流匹配策略采样 $A^{\tau+1/n}=A^\tau+\frac1n v_\pi(A^\tau,o,\tau)$。新 chunk 的前 $d$ 步无法使用（时间已过去），冻结为旧 chunk 对应步；用 ΠGDM 的梯度引导（式 2–4）：$v_{\Pi GDM}=v+\min\!\big(\beta,\frac{1-\tau}{\tau\,r_\tau^2}\big)(Y-\hat A^1)^\top\mathrm{diag}(W)\frac{\partial \hat A^1}{\partial A^\tau}$，$\hat A^1=A^\tau+(1-\tau)v$，$r_\tau^2=\frac{(1-\tau)^2}{\tau^2+(1-\tau)^2}$。**权重截断 $\beta$ 是作者加的**，少步（n=5）时否则发散（附录 A.2，取 β=5）。
- **软掩码（式 5）：** 前 $d$ 步权重 1；中间 $d\le i<H-s$ 权重指数衰减 $W_i=c_i\frac{e^{c_i}-1}{e-1}$，$c_i=\frac{H-s-i}{H-s-d+1}$；后 $s$ 步权重 0。硬掩码（只用前 $d$ 步）在小 $d$ 时明显较差；指数衰减略优于线性衰减，远优于不衰减（Fig. 8 左）；比 Diffuser 的直接覆盖式 inpainting 好。
- **Algorithm 1：** 后台线程做推理，控制线程每个 $\Delta t$ 取动作；用长度 $b$ 的延迟队列取 $\max$ 保守估计下一次 $d$；实际执行视野为 $\max(d,s_{min})$。
- **仿真（Kinetix，12 个高动态任务）：** 动作带高斯噪声，$H=8$，4 层 MLP-Mixer，每点 2048 次 rollout；延迟 0–4。RTC 在各延迟下优于 BID、朴素异步、时间集成（TE）；TE 在 $d=0$ 就差（对多模态动作取平均无效）；硬掩码 < 软掩码；RTC 的执行视野越短越好，说明它真正利用了闭环修正。
- **真机：** π0.5（H=50，$\Delta t$=20 ms）、n=5 步；基线模型延迟 76 ms，RTC 97 ms（LAN 远程推理再加 10–20 ms，约 $d\approx6$）；+100 ms、+200 ms 注入延迟分别对应 $d\approx11$、$16$。6 个任务（点蜡烛、插网线、移动做床、叠衬衫、批量叠衣、水槽洗碗）×10 次试验×3 种延迟×4 种方法=480 条，约 28 小时。**结果：** RTC 在各延迟吞吐最佳，+100/+200 ms 时显著；对注入延迟几乎无退化，同步推理线性下降；两种 TE 在 +100/+200 ms 因剧烈振荡触发保护性停机而无法运行；点蜡烛（无重试）与做床上优势最大。推理耗时（附录 Table 1–3）：移动场景总延迟 138.98 ms（模型 96.89、网络 21.20、图像缩放 11.22、其他 9.67），固定臂 108.76 ms；RTC 的去噪单步 35 ms（无 RTC 14 ms，约 2.5×），总模型 97 ms vs 76 ms；BID 在 π0.5 上延迟约 2.3 倍于 RTC。
- **超参（Table 4）：** n=5；H=8（仿真）/50（真机）；s_min=25（真机）；β=5；延迟缓冲 b=10。
- **原文自述局限：** 比直接采样带来显著额外计算（需反向传播过每个去噪步）；仅适用于扩散/流策略；更动态的场景（如腿足）只在仿真基准里，未做真机；与级联控制/MPC 的关系留作未来工作。
- **与博客/二手说法的差别：** 本归档此前“博客示例延迟约 139 ms（模型 97 ms）”与论文附录 Table 2 一致；“π0、π0-FAST、π0.5 定量结果仍是同步执行”来自博客，未在论文正文中核对。

## 本次补充核对（2026-11-14）

- **用户提供的代码链接纠正：** `https://pi.website/research/real_time_chunking` 为 PI 官方研究展示页，不是源码仓库。官方可获取的仿真源码是 `https://github.com/Physical-Intelligence/real-time-chunking-kinetix`，且同一仓亦用于后续训练期 RTC 论文（arXiv:2512.05964）。
- **核对版本与范围：** 复核 arXiv v2 HTML §3、Algorithm 1、§4、§6、Appendix A.2/A.6；阅读 PI 官方研究页与 Kinetix 仓 README；检查 `src/eval_flow.py` 的 `eval()/main()`、`src/model.py` 的 `FlowPolicy.action()/realtime_action()`、`src/train_flow.py` 的数据加载和训练入口，以及 `src/generate_data.py` 数据收集入口。此前 2026-11-07 的 v2 全文通读记录仍有效；本次是补充源码核对，不声称运行实验。
- **实验调用事实：** `eval_flow.py` 的 `RealtimeMethodConfig` 调 `FlowPolicy.realtime_action()`，在 `simulated_delay is None` 分支执行 `jax.vjp` 和 `get_prefix_weights`，再把旧块的 `inference_delay` 步与新块剩余执行步拼接，`jax.lax.scan` 驱动 Kinetix。该仓并未提供与论文真机 `InferenceLoop/GetAction` 完全相同的系统部署接口。
- **分离训练期变体：** `simulated_delay` 非空分支采用已有动作前缀和时间条件，属于后续训练期 RTC 方向；应与本文推理期 ΠGDM 引导明确区分。
