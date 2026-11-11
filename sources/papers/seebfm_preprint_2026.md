# SeeBFM: Behavior Foundation Model That Sees — 论文来源待补全

> 来源归档（ingest；论文全文尚未取得，不能当作已精读的论文）

- **标题：** SeeBFM: Behavior Foundation Model That Sees（SeeBFM：能“看”的行为基础模型）
- **类型：** 2026 preprint / under review（作者主页标注）
- **作者：** Thanh Ly、Truong-Duy Dang、Hung Nguyen、Duy M. H. Nguyen、Vien Anh Ngo、An T. Le
- **研究单位：** VinRobotics、UC San Diego、DFKI、University of Stuttgart、IMPRS-IS、CAIR at VinUniversity、IAS at TU Darmstadt（来自官方演示片头）
- **项目主页：** <https://capfab.github.io/seebfm-media/>
- **作者论文列表：** <https://anindex.github.io/>（标为 under review，未列论文 PDF/arXiv）
- **网页源文件：** <https://github.com/capfab/seebfm-media>
- **可用原文：** 官方项目演示文稿 [slides/spec.js](https://github.com/capfab/seebfm-media/blob/main/slides/spec.js) 与 [overview.vtt](https://github.com/capfab/seebfm-media/blob/main/media/overview.vtt)
- **核对范围：** 已通读官方演示的 15 段配音/分镜脚本及完整 overview.vtt，核对作者主页的状态、项目站点源码目录、公开仿真回放入口；尚未获得论文正文、数学目标、训练细节、完整实验与附录，未核对正文数值表。
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
5. **边界：** 45 cm 障碍前会停住，作者称 35 cm 以上不尝试；从“未摔倒”推断“完成穿越”是错误的。全部展示为仿真，使用深度学生的 G1 真机测试仍在筹备。
   - **wiki 映射：** [SeeBFM 实体](../../wiki/entities/paper-seebfm.md) — 局限与风险。

## 待补全文核查

- 找到作者公开的完整 PDF / arXiv / OpenReview 或正式论文版本，核实算法损失、观察频率、扫描维度、训练环境、评测表、消融、附录与模型代码开放状态。
- 原文可取得并完成方法、实验、局限的通读后，才将本归档 **通读** 改为“已完成”，并复核知识节点健康度。
