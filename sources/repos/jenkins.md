# Jenkins 源码仓库

- **URL：** <https://github.com/jenkinsci/jenkins>
- **维护者：** Jenkins 社区
- **类型：** 开源自动化服务器
- **开源状态：** 核心源码公开，仓库 README 标明 MIT 许可证；Pipeline 是一组独立插件。
- **运行入口：** `java -jar jenkins.war`
- **核查日期：** 2026-11-04
- **源码范围：** 已查看 master 分支的 [executable.Main](https://github.com/jenkinsci/jenkins/blob/master/war/src/main/java/executable/Main.java)，该入口校验 Java 版本、设置 WAR 参数、提取 Winstone 并通过反射调用其 main。尚未追踪 Pipeline 插件执行源码。
- **关联：** [Jenkins 实体](../../wiki/entities/jenkins.md)
