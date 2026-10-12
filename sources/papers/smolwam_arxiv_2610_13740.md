# SmolWAM 论文来源归档

## 元数据

- **题目：** SmolWAM: A Latent World-Action Model for Robotic Manipulation on Edge Devices
- **方法：** SmolWAM（从 Cosmos Policy 蒸馏的 0.88B 潜在世界-动作模型）
- **作者：** Timi Owolabi, Hikmah Olawore, Frances Adelakun, Ayotomiwa Oyewumi
- **单位：** ML Collective（全部作者）
- **会议：** NeurIPS 2026 Workshop on Robot Learning with World Models（据项目页）
- **预印本：** https://arxiv.org/abs/2610.13740
- **HTML：** https://arxiv.org/html/2610.13740v1
- **版本：** v1，2026-10-09，cs.RO，CC BY 4.0
- **项目页：** https://smolwam.vercel.app/（归档见 [smolwam-vercel-app.md](../sites/smolwam-vercel-app.md)）
- **代码：** https://github.com/iitimii/SmolWAM（归档见 [smolwam.md](../repos/smolwam.md)）；权重 https://huggingface.co/timiiowolabi/SmolWAM
- **入库日期：** 2026-10-12
- **读取范围：** arXiv HTML 正文与表格（WebFetch 抽取）；附录未逐页核对。

## 论文要解决的问题

世界-动作模型（WAM）同时预测动作与动作应产生的观测，最强的 WAM 用视频扩散 Transformer，边缘设备难以承担；高效策略（SmolVLA、TurboVLA）通常去掉世界模型，Fast-WAM 在测试时跳过未来预测。作者问：能否把强预训练 WAM 做小，同时仍预测未来。

## 教师：Cosmos Policy

基于 Cosmos-Predict2，DiT 28 块、1.96B 参数；对 9 个潜帧（每帧 196 token）去噪：空白、状态、腕部与第三视角相机潜变量、H=16 动作块、未来状态/相机潜变量/价值；4.8B T5-XXL 文本编码器（512 嵌入，交叉注意力）、Wan2.1 VAE（33 帧片段）；rectified flow，发布管线 5 步；LIBERO 成功率 98.5%（引自原论文）；每 16 动作块在 A100（缓存文本）约 624 ms、42.6 TFLOP、7.55 GiB。

## 方法：如何得到 SmolWAM

1. **单帧视觉潜变量：** 每路图像仅用 Wan2.1 编码器（54M）编码一次，0.5M 参数适配器映射到片段模式潜变量；无适配器时教师动作 L1 从 0.0089 升到 0.038，有则回到 0.0092。
2. **缓存 BERT 文本并零填充保留注意力 sink：** BERT-base 特征缓存后投影并零填充到 512 位；直接截到 32 或 128 位会使教师闭环成功降到 0/5，解析恢复 softmax 质量后回到 5/5；加入 TurboVLA 的 6 个零初始化 vision→language 层。
3. **保边界的深度剪枝：** 按 block influence 评分，保留块 0–2 与 19–27，28 块中留 12 块（0.85B）；去掉块 1、26 或 27 筛选全败；同时去掉 12–15 四个近恒等块也全败，说明剪枝影响不可叠加。
4. **状态条件动作查询：** 零初始化 MLP 把机器人状态加到 28 个动作查询上，取代 196 token 动作帧；总上下文 1,008 token（教师 1,764）。
5. **一次前向读出：** 在 τmax≈0.988 固定噪声下评估一次主干，从一步干净估计读动作与未来潜变量；未来潜变量构成训练时受监督、推理时不解码的“世界头”。

## 训练

- 数据：Cosmos Policy 发布的 1,700 条成功 LIBERO 示范，每 4 步抽样得 68,830 条观测。
- 目标：教师在固定噪声下的一步预测（动作块与未来潜变量），缓存耗时 66 min（1×A100）。
- 损失：学生对示范与教师的 L1，外加 0.1 权重的状态/价值帧项；课程：5,000 步中前 1,000 步用教师 1,764 token 布局，之后紧凑布局。
- AdamW，继承权重 lr 2e-5、新增权重 1e-4，batch 32，bf16；共 3.6 GPU 小时（1×A100）。

## 评测设置与结果

- LIBERO 四套件（Spatial/Object/Goal/Long，各 10 任务）；完整协议每任务 10 次共 400 回合，种子 195，16 动作块开环执行；筛选集 60 回合（每套件 3 任务 × 5 次）。延迟为 A100-SXM4-40GB、batch 1、bf16 下从观测到动作块的墙钟时间（含预处理与 VAE）。
- **Table 1：** Cosmos Policy（5 步）1.96B / 42.6 TFLOP / 624 ms / 7.55 GiB / 筛选 96.7 / 完整 98.2†；SmolWAM 0.88B / 2.30 / 57 / 2.36 / 93.3 / 95.5；SmolWAM+7.5k 步 94.2；INT8 权重 筛选 91.7（延迟 65 ms 为模拟）；8 块 0.60B 筛选 83.3；SmolWAM-LA（4 步）筛选 53.3；剪枝+LoRA 蒸馏 1.40B 完整 87.5†。† 来自较早 L4 评测管线。
- **分套件（完整）：** SmolWAM Spatial 99.0、Object 100.0、Goal 94.0、Long 89.0，总 95.5（382/400，Wilson 区间 93.0–97.1）；教师 97.0/100.0/99.0/97.0，总 98.2（393/400）。失败多为完成首个子目标后卡在第二个。
- **速度/显存：** 57 ms 中 Transformer 37 ms、VAE 编码 13 ms、预处理 7 ms；约 11× 更快、18.5× 更少 FLOPs、3.2× 更少显存；吞吐约 17.7 chunk/s。教师在 8/16 动作后重查询，筛选 0 回合成功。INT8 逐通道量化 172 个线性层，存储 1.64→0.85 GiB，筛选少 1 回合，延迟未降。
- **世界头：** 96 个留出上下文，前瞻 16 步潜变量 L1 0.101（复制当前帧 0.133，低 24%）；RAFT 光流端点误差 1.43 px（复制 1.63）；解码预测保留场景、模糊外观，LPIPS 与复制帧同级（0.128 vs 0.115）。
- **更丰富 WAM 组件：** SmolWAM-LA（+39M 参数，潜动作 token、MoT 塔、4 步流匹配、“dream forcing”）筛选 53.3%，239 ms；对世界头做奖励加权微调使成功率塌到 5.0–16.7%。
- **消融：** 单标志消融在 2,500 步（主实验一半）均在参考（78.3%）5 回合内，筛选集无法区分；6 项计划消融（含去世界头、去蒸馏目标、去布局课程）未做。

## 复现与解释边界

- 作者自述：仅 LIBERO，无真机；缩减深度对长时域（LIBERO-Long）伤害更大；离线验证 L1（约 0.09，约教师一步误差 10 倍）不是闭环成功的好代理；块开环限制反应性。
- 教师 98.2% 与 LoRA 基线 87.5% 来自较早 L4 管线，而 SmolWAM 延迟为 A100，跨硬件比较需谨慎；教师 98.5% 引自原论文而非复现。
- INT8 延迟用 bf16 内核模拟；端侧实机测量未做。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-smolwam.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 周报来源：[wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md](../blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md)

## 一手来源

- <https://arxiv.org/abs/2610.13740>
- <https://arxiv.org/html/2610.13740v1>
- <https://smolwam.vercel.app/>
- <https://github.com/iitimii/SmolWAM>
