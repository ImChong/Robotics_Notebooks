# 具身智能小站 10-29 期 58 篇盘点通读（原文通读记录）

> 来源归档（原文通读 · 支撑 `wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md`）

- **标题：** 机器人缺的，是更多示范，还是更多成功解法？｜附代码与项目入口
- **作者：** 具身智能小站（微信公众号）
- **类型：** blog
- **原文：** <https://mp.weixin.qq.com/s/s23oOBojr5YbchuFmQhcwA>；抓取全文落盘于 [`sources/raw/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md`](../raw/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md)
- **一句话说明：** 一期 58 篇具身论文的优先级盘点：1 篇深读（DiscoDemo）、2 篇跟进（OpenTac、Watch Your Step）、55 篇每篇一个判断的扫读。
- **核对范围：** 通读抓取全文：阅读路线、深读与两篇跟进的"看点/结果"、第 04–58 篇的一行判断与资源链接；各篇论文本身（除深读的 DiscoDemo，见 `sources/papers/discodemo_reading_arxiv_2610_32093_es58a.md`）未逐篇读原文，页面中各节点数字为公众号转述。
- **通读：** 已完成

## 核心摘录（均出自原文）

- 阅读路线按优先级分三档：1 篇深读、2 篇跟进、55 篇扫读；每篇扫读只保留一个判断，并附论文/代码/项目页入口。
- 深读 DiscoDemo：把 RL 当示范生成器，逆向课程 + 成功门控多样性 + 安全筛选；固定每任务 3000 条示范与同一 π0.5，真机四任务平均成功率 77.5% 对 38.75%（P-RFCL）。
- 跟进 OpenTac：同一传感器描述文件连接制造、采集、仿真与跨皮肤转换；跨皮肤实验测的是接触形状迁移。
- 跟进 Watch Your Step：由本体状态与速度指令预测未来两个落脚点，只读取邻域高程图小块；可能漏掉侧方接近的障碍。
- 扫读各篇多附带"需注意"类限定语（如数字来自特定配置、仿真为主、依赖理论假设），页面据此保留了开源状态与证据边界。

## 对 wiki 的映射

- [`wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md`](../../wiki/overview/embodied-station-2026-10-29-58-papers-technology-map.md)
