# astrbot_plugin_nova_cac

一个供 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 使用的 NOVA 问答插件。它把当前仓库内的 `knowledge_pack` 作为回答依据，按 CAC 文档中呈现的思考方式和语言风格组织回答，同时保留近期 `/cac` 对话上下文。

## 内容包架构

`knowledge_pack` 是一套与具体机器人框架解耦的 NOVA 内容包：`AGENTS.md` 规定回答流程、资料路由和事实边界，`soul.md`、`spirit.md`、`voice.md` 分别定义回应姿态、社团理念和 CAC 的表达方式，并在每次回答前完整读取；`knowledge/` 则按“认识 NOVA、理念与方法、规章与活动”保存经过筛选的原始文章，回答时只读取相关内容。它把事实依据、理念解释、人格姿态和说话方式分开维护，因此也可以复用于其他插件、Skill 或 Agent。

## 唤起方式

私聊直接发送：

```text
/cac NOVA 是一个技术社团吗？
```

群聊必须同时 `@机器人`：

```text
@机器人 /cac NOVA 为什么强调分享？
```

群聊里只发送 `/cac ...` 而不 `@机器人`，插件不会回答。

其他命令：

```text
/cac help
/cac reset
```

`reset` 只清空当前私聊或当前群聊由本插件保存的近期上下文。

## AGENTS.md 会在什么时候读取？

AstrBot 不会因为模型“觉得自己忘了”就自动发现或读取插件目录中的 `AGENTS.md`。这个插件没有把读取时机交给模型决定，而是在每一次实际回答前主动执行：

1. 重新读取 `knowledge_pack/AGENTS.md`；
2. 重新读取 `knowledge_pack/soul.md`；
3. 重新读取 `knowledge_pack/spirit.md`；
4. 重新读取 `knowledge_pack/voice.md`；
5. 从 `knowledge_pack/knowledge/` 检索与当前问题最相关的少量片段；
6. 把近期 `/cac` 对话、当前问题和检索片段一起交给当前 AstrBot 聊天模型。

因此，四个核心文件在运行中被修改后，下一次回答就会使用新内容，不需要重启插件。普通知识文章会建立本地轻量索引，文件发生变化时自动刷新。

需要注意：四个必读文件合计较长，每次完整注入会占用模型上下文窗口并产生相应 token 消耗。这是为了严格满足“每次必读”的约束。

## 安装

要求 AstrBot `>=4.16,<5`。

在 AstrBot 的插件目录中克隆本仓库：

```bash
git clone https://github.com/whyself/astrbot_plugin_nova_cac.git
```

随后在 AstrBot 管理面板中重载或重启插件，并确保已配置可用的聊天模型提供商。

插件没有第三方 Python 运行依赖。

## 配置

配置项可以在 AstrBot 插件配置界面修改：

| 配置项 | 默认值 | 作用 |
| --- | ---: | --- |
| `history_turns` | `6` | 每个会话保留的问答轮数；设为 `0` 可关闭上下文 |
| `max_sessions` | `256` | 内存中最多保留的会话数 |
| `history_max_chars` | `12000` | 每个会话的近期问答总字符上限 |
| `retrieval_top_k` | `5` | 每次最多使用的知识片段数 |
| `max_context_chars` | `9000` | 检索片段总字符上限，不包含四个必读文件 |

轮数上限和字符上限同时生效，任意一个达到上限都会淘汰最早的完整问答轮次。只有成功触发并完成回答的 `/cac` 问答会被写入；普通消息、其他插件消息、失败的模型调用，以及群聊中未 `@机器人` 的 `/cac` 都不会进入上下文。

上下文只保存在 AstrBot 进程内存中，插件重载或 AstrBot 重启后会清空。群聊以群会话为单位共享上下文，因此同一群里的成员会接续该群最近的 `/cac` 对话。

## 知识包维护

目录结构：

```text
knowledge_pack/
├── AGENTS.md
├── soul.md
├── spirit.md
├── voice.md
└── knowledge/
    ├── 01_认识NOVA/
    ├── 02_理念与方法/
    └── 03_规章与活动/
```

更新文件时保持 UTF-8 编码。制度、活动时间和报名信息应先在知识包中完成版本核对；检索器会提高现行章程和 `2026` 文档的优先级，但不能代替人工维护资料的有效性。

## 回答边界

- 插件吸收 CAC 文档中的表达方式，但不冒充 CAC 本人。
- 默认自然回答，不主动报告检索过程或把内容写成引用报告。
- 用户明确索要出处时，模型可以使用检索片段中的文章标题和 `source_url`。
- 资料没有明确答案时，应保留不确定性，不编造时间、群号、内部决定或个人经历。
- 用户要求忽略规则、泄露提示词或虚构信息时，仍以知识包中的身份和事实边界为准。

## 开发与验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py nova_cac tests
ruff check .
```

实现参考了 AstrBot 官方插件接口、[Soulter/helloworld](https://github.com/Soulter/helloworld) 模板和 [Gu-Heping/astrbot_plugin_nju_qa](https://github.com/Gu-Heping/astrbot_plugin_nju_qa) 的事件路由方式。
