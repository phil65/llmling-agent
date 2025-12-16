---
title: Execution environments
description: Event handler setup and configuration
icon: material/bell-ring
---

Execution environments allow you to configure the runtime environment for your agent. It's where code is run and where processes are managed / commands are executed.

Any Agent which can perform IO (regular Agents & ACP Agents) can get assigned an execution environment.

/// mknodes
{{ "exxec.configs.ExecutionEnvironmentConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

Theres one more execution environment, the ACP environment.
This one cannot get assigned manually, but it becomes to default execution environment for any agent participating in an ACP session. (its overridable though, an ACP agent can also work remotely!)
