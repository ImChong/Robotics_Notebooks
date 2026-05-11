# Linking Rules

与「内容应放进仓库哪个根目录」相关的取舍，见 [content-directories.md](content-directories.md)。**Schema 目录总索引**：[README.md](README.md)。

## 目标

知识库底层应是图结构，而不是单树结构。

## 基本原则

1. 页面之间尽量显式链接
2. 同一概念可以被多个页面引用
3. 不要求一个页面只有一个父节点
4. 目录只是组织方式，不代表唯一语义归属

## 推荐链接方式

使用相对路径 markdown 链接：

```md
- [Sim2Real](../concepts/sim2real.md)
- [Reinforcement Learning](../methods/reinforcement-learning.md)
```

## 页面内建议保留的关联区块

推荐在每页末尾加入：
- 关联页面
- 推荐继续阅读

示例：
```md
## 关联页面
- [Locomotion](../tasks/locomotion.md)
- [Imitation Learning](../methods/imitation-learning.md)

## 推荐继续阅读
- [Humanoid Control Roadmap](../roadmaps/humanoid-control-roadmap.md)
```

## source 到 wiki 的关系

- `sources/` 中的资料不要求互相强链接
- `wiki/` 中的知识页必须尽量链接到相关页
- `wiki/` 可以引用 `sources/`，但不能退化成纯链接堆
