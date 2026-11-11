# HARP-Bench: Benchmarking Robot Manipulation Around Active Humans（arXiv:2611.11123）

> 来源归档（paper）

- **标题：** HARP-Bench: Benchmarking Robot Manipulation Around Active Humans
- **类型：** paper（仿真基准 + 数据生成管线）
- **arXiv：** <https://arxiv.org/abs/2611.11123>
- **PDF：** <https://arxiv.org/pdf/2611.11123v1>（v1，2026-11-08，cs.RO，21 页；CoRL 2026 稿，首页标注 10th Conference on Robot Learning）
- **作者与机构：** Qinhong Zhou、Riley Zong（共同一作）、Chunru Lin、Diyi Hu、Shivraj Singh Bhatti、Hongxin Zhang、Lin Qiu、Hanqing Zeng、Chuang Gan；马萨诸塞大学阿默斯特分校、明尼苏达大学双城分校、Meta、独立研究者
- **代码：** https://github.com/UMass-Embodied-AGI/harp
- **项目页：** https://harp-bench.vercel.app/
- **入库日期：** 2026-11-11
- **一句话说明：** 把「会动的人」作为操作问题的一部分：37 个仿真任务按 Cueing / Contribution / Interruption / Co-presence 四类人类行为分组，闭环评测任务成功率与碰撞 / 距离类安全指标，并提供脚本示教生成与 LeRobot 兼容数据接口。

## 核对范围与通读

- **核对范围：** 论文 v1 PDF 全部 21 页：正文 §1–§6（引言、相关工作、§3 任务设计 / 闭环评测 / 数据生成、§4 实验含表 2–5 与图 3、结论、局限）、参考文献、附录 A（头像渲染开销，表 A1）、附录 B（37 个任务描述，表 A2）、附录 C（基准统计：147 个物体资产、46 个头像外观、动作与事件触发逻辑，表 A3、图 A1–A3 的图注）；仓库 README（GitHub raw，静态阅读）与项目页 HTML 的文字部分仅作开源状态核对，未逐页读仓库 docs/，未运行任何代码，图中像素细节（图 1–3、A3）只按图注与正文理解。
- **通读：** 已完成

## 开源状态（步骤 2.5）

- **已开源**：论文摘要给出仓库 https://github.com/UMass-Embodied-AGI/harp；README 静态阅读：Genesis 仿真（`genesis-world==1.2.0`，NYX 渲染器，需 Vulkan GPU，A100/H100 不能跑 NYX），资产包在 Hugging Face 数据集 `zhouqh/harp-bench-assets`，基线检查点为 `zimplex/harp-vla-train37-{act,diffusion,pi0,pi05,smolvla}`。README 称 LeRobot 训练数据集不作为下载发布，需自行用脚本专家采集。
- 项目页 https://harp-bench.vercel.app/ 可访问，只读了页面文字与结构，未逐项核对其表格。

## 核心摘录（均出自论文，带章节）

1. **基准规模（§3、§4.1）：** 37 个任务 = 7 Cueing + 10 Contribution + 10 Interruption + 10 Co-presence；这是「任务 - 条件实例」，同一底层操作技能有多个人类变体，论文明说不代表 37 种不同技能（§3.1）。
2. **人类建模（§3.2、附录 C.2）：**46 个头像外观继承自 Virtual Community； 蒙皮头像由 Mixamo 与 AMASS 衍生动作驱动；人侧事件是脚本化的，时机可随机（Contribution 早期随机时刻开始、Interruption 随夹爪位置触发、Co-presence 随机选无关动作）；机器人侧为物理仿真。
3. **指标（§3.2）：** 任务成功（含「机器人 - 头像碰撞即失败」）、Collision、Min. Dist.、Violat.（距离小于 τ=0.10 m 的时间步比例）。论文声明 10 cm 是评测约定，不是 ISO/TS 15066:2016 规定的防护距离，也不证明真实安全合规。
4. **数据与划分（§4.1）：** 训练 11,100 条脚本示教（37 任务 × 300）；每任务留出 100 个评测种子。
5. **主结果（§4.2、表 2、表 3）：** 全状态专家 0.99、无意图专家 0.48；学习策略 π0 0.19、ACT 0.10、DP 0.09、SmolVLA 0.06。安全：ACT 三项最好（Violat. 0.080、Min. Dist. 0.133 m、Collision 0.102），π0 成功率最高但 Collision 0.207、Min. Dist. 0.095 m。
6. **人类变体数据的效果（§4.2、表 4）：** 逐任务训练的 ACT，25 任务子集共 7,500 条示教；人类变体平均 0.10→0.15，基础任务 0.42→0.41。
7. **失败分析（§4.2、图 3）：** 学习策略失败中「操作 / 恢复」占 60–66%，碰撞占 22–27%（另有「意图」类，数值正文未给）。
8. **真机（§4.3、表 5）：** xArm 桌面，dump bin 任务族，ACT，每设置 30 次。基础策略：基础任务 0.40、Contribution 0.13、Interruption 0.00；用 HARP-Bench 的 Contribution 示教训练后 Contribution 0.13→0.33，Interruption 仍为 0.00。作者自述该研究不验证全部四类条件或真实安全。
9. **渲染开销（附录 A、表 A1）：** A16 GPU、Nyx、640×480：无头像 203.01 FPS、静态头像 98.95、动画头像 89.55；论文称 20 Hz 下约 4.5 倍实时。
10. **作者承认的局限（§6）：** 脚本化头像无持续的人机适应；以桌面操作为主；无移动操作和长程协作；感知与接触动力学简化；成功率与几何安全指标不直接衡量等待、绕行、识别人完成的子任务或社交舒适度；仿真安全不等于部署安全。

## 与仓库 README 的出入（静态读，未运行）

- README 称发布的评测集 `eval_v3_sampled` 含 3,403 个「专家可解」种子；论文 §4.1 说每任务 100 个评测种子（即约 3,700）。两者口径可能不同，原因论文未写，未核实。
- README 称发布 `pi05` 检查点，论文 §4.1 基线只列 ACT、DP、π0、SmolVLA。
- README 称仓库含 49 个规范任务与六种机器人形态，其中 37 个为论文评测任务；论文正文没有交代仿真所用机器人型号（真机为 xArm）。

## 对 wiki 的映射

- [paper-harp-bench-active-humans](../../wiki/entities/paper-harp-bench-active-humans.md)
