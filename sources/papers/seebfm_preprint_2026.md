# SeeBFM: Behavior Foundation Model That Sees — 论文来源待补全

> 来源归档（ingest；论文全文尚未取得，不能当作已精读的论文）

- **标题：** SeeBFM: Behavior Foundation Model That Sees（SeeBFM：能“看”的行为基础模型）
- **类型：** 2026 preprint / under review（作者主页标注）
- **作者：** Thanh Ly、Truong-Duy Dang、Hung Nguyen、Duy M. H. Nguyen、Vien Anh Ngo、An T. Le
- **研究单位：** VinRobotics、UC San Diego、DFKI、University of Stuttgart、IMPRS-IS、CAIR at VinUniversity、IAS at TU Darmstadt（来自官方演示片头）
- **项目主页：** <https://capfab.github.io/seebfm-media/>
- **作者论文列表：** <https://anindex.github.io/>（标为 under review，未列论文 PDF/arXiv）；[官方项目页](https://capfab.github.io/seebfm-media/) 明确写 `arXiv (coming soon)`
- **网页源文件：** <https://github.com/capfab/seebfm-media>
- **可用原文：** 官方项目演示文稿 [slides/spec.js](https://github.com/capfab/seebfm-media/blob/main/slides/spec.js) 与 [overview.vtt](https://github.com/capfab/seebfm-media/blob/main/media/overview.vtt)
- **核对范围：** 已逐节通读作者官方项目页（How it works、Results、Compared with P-BFM、What we learned、Cameras、Limitations、Credits）、逐项核对两组结果表和作者主动报告的限制；通读官方演示的 15 段配音/分镜脚本及完整 overview.vtt、网页预录轨迹回放主要源码。**论文本体尚未公开**，故无法通读论文正文、附录及最终实验表。详细项目页笔记见 [项目网页通读记录](../sites/seebfm-project-page.md)。
- **通读：** 未完成
- **核对日期：** 2026-11-11

## 已核实的技术线索（以官方演示为依据，不等同全文）

1. **意图与地形分离：** 从 BFM-Zero 出发，使用高度扫描（terrain height scan）继续做跨地形后训练；reward / goal pose / motion 的 prompt 仍代表同一行为意图。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 表征接口和网络条件化。
2. **扫描进入哪条网络：** 官方脚本称除 backward network 外，各网络读取高度扫描；backward 网络将奖励、目标状态与动作转为行为 prompt，因此不使 prompt 语义随地形改变。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 核心原理和消融意义。
3. **部署感知替代：** 独立 depth student 从两个深度相机预测高度扫描；以真实扫描引导的策略轨迹为目标训练学生，部署时替换扫描输入，冻结原策略而非重新训练。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 训练阶段/摄像头阶段。
4. **仿真指标：** 官方视频/分镜对六个 locomotion prompts 报告 SeeBFM 上楼梯 54.4%、箱体 41.0%、下楼梯 68% 的 crossing；视频对应 BFM-Zero 为 0%、0%、5%。上楼梯无有效高度差输入/不使用扫描训练的消融分别为 4.0%/0%。这些数字目前只能以官方演示口径记录，不能替代对论文表格、seed 与统计协议的精读。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 实验与评测。
5. **边界：** 官方项目页称高于约 25 cm 方块无法跨过；官方视频另示 45 cm 方块前停住，称高于 35 cm 不尝试（量法和阈值不同，不能自行统一）；从“未摔倒”推断“完成穿越”是错误的。全部展示为仿真，使用深度学生的 G1 真机测试仍在筹备。\n   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 局限与风险。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 局限与风险。

## 最新项目页逐节核对补充（仍非论文全文）

- [How it works / What we learned](https://capfab.github.io/seebfm-media/)：598 点地形高度扫描，97.7% 下一落脚点覆盖；379 h 总动作素材、78 h 地形运动；让 backward 网络看到扫描会使作者报出的地形跟踪误差从 1.14 增至 1.31；移除 prompt 网络平均项缓解同一行为偏置。
- [Compared with P-BFM](https://capfab.github.io/seebfm-media/)：0.10–0.15 m 台阶、训练扰动关闭条件，分别报告 0.7 m/s forward 单指令与四种 walking prompt 的跨越/跌倒；含 SeeBFM height scan、camera student、P-BFM prompt-only、P-BFM with-command 四组。**官网明确实验在不同模拟器，不能当成严格同分布基准**；全表已归档至 [项目页来源](../sites/seebfm-project-page.md)。
- [Cameras / Limitations](https://capfab.github.io/seebfm-media/)：深度学生相对真实扫描让四种行走提示的跌倒率上升；上下楼梯参考轨迹中已录制 climbing 只成功跟踪 68/257；真机验证仍未完成。
- [Playground](https://capfab.github.io/seebfm-media/playground/)：作者声明可回放 19,232 次录制仿真尝试；演示不等于策略网络在浏览器里在线推理。

## 待补全文核查

- 找到作者公开的完整 PDF / arXiv / OpenReview 或正式论文版本，核实算法损失、观察频率、扫描维度、训练环境、评测表、消融、附录与模型代码开放状态。
- 原文可取得并完成方法、实验、局限的通读后，才将本归档 **通读** 改为“已完成”，并复核知识节点健康度。
