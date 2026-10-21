# AIGeeksGroup/HumanoidTTT

> 来源归档

- **标题：** HumanoidTTT: Test-Time Capability Reuse for Efficient Humanoid Control
- **类型：** repo
- **来源：** 北京大学计算机学院（Jingtai Yang、Yining Wu、Yanjun Li、Zeyu Zhang、Hao Tang）
- **链接：** https://github.com/AIGeeksGroup/HumanoidTTT
- **论文：** https://arxiv.org/abs/2610.00198
- **项目页：** https://aigeeksgroup.github.io/HumanoidTTT/
- **权重：** https://huggingface.co/AIGeeksGroup/HumanoidTTT
- **入库日期：** 2026-10-16
- **一句话说明：** 为人形机器人提供测试时完整动作复用与有限容量能力整合的 Python 方法组件和初始策略权重。

## 开源状态与运行边界

代码仓库公开，README 提供 Python 3.10+ 安装步骤（pip install -e .）、组件用法和代码目录说明。Hugging Face 提供初始 consolidation checkpoint、配置与校验和。仓库根目录本次检查未见 LICENSE 文件，代码开放可读但许可条件尚需确认。

项目组件包括：

| 模块 | 用途 |
|---|---|
| entry_features.py | 生成 45 维入口状态特征 |
| certificates.py | 构建和判断动作入口适用性证书 |
| consolidation.py | Store 动作评分器与 Double-DQN 更新 |
| motion_features.py | 16 维动作特征 |
| memory.py | 存储动作和局部探测辅助函数 |
| signatures.py | 请求与运行时兼容签名 |
| types.py | 运动参考和 Store 条目接口 |
| hashing.py | 工件哈希 |

README 明确说明：应用侧需提供生成器、HoloMotion 运行时、动作档案、入口资格观察及执行循环；生成器和 tracker checkpoint 依赖上游 [OMG](https://github.com/Tsinghua-MARS-Lab/OMG) 与 [HoloMotion](https://github.com/HorizonRobotics/HoloMotion)。因此这是部分开源的方法组件，不是开箱即用的完整机器人部署栈。

## 与仓库内条目的关系

- [HumanoidTTT 论文实体](../../wiki/entities/paper-humanoidttt.md)
- [项目页归档](../sites/humanoidttt-project.md)
- [论文原始来源](../papers/humanoidttt_arxiv_2610_00198.md)

## 一手入口

- [README](https://github.com/AIGeeksGroup/HumanoidTTT#readme)
- [模型权重](https://huggingface.co/AIGeeksGroup/HumanoidTTT)
- [项目主页](https://aigeeksgroup.github.io/HumanoidTTT/)
