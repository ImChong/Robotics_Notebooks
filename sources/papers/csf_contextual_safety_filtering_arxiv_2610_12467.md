# CSF: Contextual Safety Filtering for Motion Generators 论文归档

## 元数据

- **题目：** CSF: Contextual Safety Filtering for Motion Generators
- **作者：** Lizhi Yang, Yiling Hou, Yao Tang, Junheng Li, Daniel Weng, Blake Werner, Aaron D. Ames
- **单位：** California Institute of Technology（Lizhi Yang、Yiling Hou、Yao Tang、Junheng Li、Blake Werner、Aaron D. Ames）；New York University（Daniel Weng）
- **预印本：** https://arxiv.org/abs/2610.12467
- **HTML：** https://arxiv.org/html/2610.12467v1
- **版本：** v1，2026-10-08，cs.RO / cs.LG
- **项目页：** https://lzyang2000.github.io/csf/
- **官方代码：** https://github.com/lzyang2000/csf（Apache-2.0；第三方生成器与权重遵循各自许可证）
- **硬件演示：** Unitree G1；ARDY 生成 + CSF 过滤 + 预训练 SONIC 全身跟踪策略

## 问题与核心思路

文本动作生成器能产生平滑、可跟踪的全身动作，却无法仅凭动作本身判断它在当前场景里是否合适。同一条“向前出拳”可能是对着箱子执行，也可能在人进入视野后指向人。只检查语言提示无法处理误导性提示，几何碰撞约束也难以表达这种语义安全差异。

CSF（Contextual Safety Filtering）在推理阶段冻结原生成器，不额外训练安全分类器。它为每条自然语言安全规则生成“危险动作参考”和“安全替代动作参考”，用两者差值定义动作空间中的仿射安全边界；再根据指令与视觉场景实体门控规则，并在生成采样期间以 CBF-QP 修正当前输出估计。

## 方法

1. **规则与参考：** 每条规则指定不允许的行为及其受保护实体，例如“不要击打人”。同一生成器分别生成危险参考与安全替代参考；安全替代从中性动作库选取，必要时回退到 neutral stand。
2. **语义边界：** 危险参考减去安全参考形成规则的语义方向。当前动作估计沿该方向的投影构成仿射 margin；margin 非负表示估计位于论文定义的安全半空间。
3. **场景门控：** 生成器的文本编码器比较用户命令和感知到的场景实体标签。命令提到受保护对象，或场景检测到该对象时，对应规则被激活；没有激活规则时，动作原样通过。
4. **采样内 QP：** 每个采样步都把过滤后的输出估计送回原生成器的更新。QP 在“接近原估计”和“朝安全参考拉回”之间平衡；γ=0 时，合规候选不变，违规候选只做最小范数修正。
5. **架构适配：** 统一规则、margin、门控和 QP；适配器读取 sampler 当前最终动作估计，适配 motion feature、VAE latent 或自回归未提交窗口。可分离时排除 root channels，避免靠平移角色来满足规则。
6. **执行期安全盾：** 场景若在执行中变化，检查尚未执行的动作窗口；保留已执行部分，从短暂响应缓冲帧起将剩余轨迹接到对齐的安全延续，再交回原跟踪策略。

## 实验与结果

- **四类生成器：** Kimodo-G1（DDIM diffusion，417 维 motion feature）、ECHO（DPM-Solver，38 维 feature）、MotionHiFlow（VAE latent flow matching）、ARDY（windowed autoregressive hybrid tokens）。
- **规则范围：** 六类规则，其中四类与对人暴力动作有关，另含儿童与动物保护规则；基准含 28 组危险—良性命令对。
- **规模：** 论文报告每个 backbone 672 段、总计 2,688 段生成动作，过滤与未过滤各 1,344 段。
- **危险事件：** 三个 backbone 的危险事件率降至未过滤条件的 0.1 倍，ECHO 为 0.2 倍；即相对下降 90% / 80%，不是危险事件归零。
- **良性动作保留：** 四种 backbone 保留 88–100% 的良性动作（原样通过或接近指定安全替代动作）。各模型均优于向站立姿态简单混合的参照，但保留代价随表示空间变化。
- **推理开销：** RTX 4090 上，1–6 条激活规则的 rule-space dual 为 0.01–0.18 ms，完整 filter call 为 0.3–1.2 ms。ECHO、MotionHiFlow、ARDY 参考动作缓存初始化 0.6–0.9 s，逐样本开销最高约 19%；Kimodo 每步构造参考，使 4 秒动作生成由 1.61 s 增至 7.60 s，其中 QP 约 0.14 s。
- **真机：** G1 上使用 ARDY 生成、CSF 过滤、预训练 SONIC 跟踪。展示显式对人出拳、提示目标为柱子但场景有人、仅有柱子的良性动作，以及执行中有人进入视野；运行时盾保留已执行部分并改写剩余参考。

## 边界与复现注意

- 理论约束的是生成器输出估计上的仿射 margin；语义是否正确仍依赖规则、文本—实体门控和场景感知，不等同于对人体伤害或碰撞概率的独立认证。
- 规则集覆盖六类危险概念；更广泛的行为、关系和上下文仍需设计并验证安全参考与激活条件。
- CSF 过滤生成结果，部分 backbone 会排除 root channels；这不等同于主动规划机器人远离危险对象。
- 不同生成器的 prediction space 和 margin 尺度不同，不应跨模型比较 margin 数值。
- 论文未来工作包括扩展 root trajectory、规则触发时规划移动方向，以及依据变化场景生成更长的安全延续。
- 官方仓库 README 提供 scripts/serve.sh 浏览器 demo 启动方式：Linux、NVIDIA GPU（24 GB）、uv；模型使用需 Hugging Face 登录并接受 Llama 3 许可。第三方生成器与权重遵循各自许可证。README 的 quick start 面向交互式 demo；论文 G1 部署的完整复现还需核实仓库配置、依赖模型与机器人部署条件。

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12467>
- HTML 全文：<https://arxiv.org/html/2610.12467v1>
- 官方项目页：<https://lzyang2000.github.io/csf/>
- 官方代码：<https://github.com/lzyang2000/csf>
