# astrbot_plugin_nova_cac

一个供 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 使用的 NOVA 问答插件。它把当前仓库内的 `knowledge_pack` 作为回答依据，让 Agent 自行使用向量检索、关键词搜索和原文读取工具完成取证，再按 CAC 文档中呈现的思考方式和语言风格组织回答，同时保留近期 `/cac` 对话上下文。

## 内容包架构

`knowledge_pack` 是一套与具体机器人框架解耦的 NOVA 内容包：`AGENTS.md` 规定回答流程、资料路由和事实边界，`soul.md`、`spirit.md`、`voice.md` 分别定义 CAC 的人格与思考方式、NOVA 的社团理念和 CAC 的表达方式，并在每次回答前完整读取；`knowledge/` 则按“认识 NOVA、理念与方法、规章与活动”保存经过筛选的原始文章，回答时只读取相关内容。它把事实依据、理念解释、人格和表达方式分开维护，因此也可以复用于其他插件、Skill 或 Agent。

## 唤起方式

群聊和私聊都直接发送：

```text
/cac NOVA 是一个技术社团吗？
```

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
5. 让研究 Agent 根据问题自行选择混合检索、关键词搜索、文章列表、目录和原文读取工具；
6. 只把研究阶段实际读过的原文证据交给无工具的回答阶段，再结合近期 `/cac` 对话和当前问题生成回答。

因此，四个核心文件在运行中被修改后，下一次回答就会使用新内容，不需要重启插件。普通知识文章会建立本地 SQLite、BM25 和 Chroma 索引，文件发生变化时自动刷新。

需要注意：四个必读文件合计较长，每次完整注入会占用模型上下文窗口并产生相应 token 消耗。这是为了严格满足“每次必读”的约束。

## 安装

要求 AstrBot `>=4.16,<5`。

在 AstrBot 的插件目录中克隆本仓库：

```bash
git clone https://github.com/whyself/astrbot_plugin_nova_cac.git
```

随后在 AstrBot 管理面板中重载或重启插件，并确保已配置可用的聊天模型提供商。若要启用向量检索，还需在 AstrBot 中配置 Embedding Provider；没有嵌入提供商时，插件会自动退化为关键词检索，仍可回答。

插件通过 `requirements.txt` 安装 ChromaDB，用来保存本地向量索引。

## Agent 检索流程

回答分为两个阶段。研究阶段拥有以下本地工具，并由模型按问题自行决定调用顺序：

- `search_knowledge_base`：融合 Chroma 向量相似度与 BM25 关键词得分，返回候选片段；
- `grep_local_docs`：按明确的名称、日期、条款或关键词定位 Markdown 行；
- `list_docs`、`get_doc_outline`：浏览文章和标题结构；
- `read_doc`：读取指定文章的准确行段，并把它登记为可用于回答的证据。

搜索结果只是候选，不能直接进入最终回答。研究 Agent 必须再用 `read_doc` 读到正文；随后回答阶段关闭全部工具，只接收已登记的原文证据。内部证据标记在输出前会校验并移除。用户没有主动问“来源、出处、原文、链接”时，最终回答仍保持自然口语，不展示检索报告或引用列表。

## 配置

配置项可以在 AstrBot 插件配置界面修改：

| 配置项 | 默认值 | 作用 |
| --- | ---: | --- |
| `history_turns` | `6` | 每个会话保留的问答轮数；设为 `0` 可关闭上下文 |
| `max_sessions` | `256` | 内存中最多保留的会话数 |
| `history_max_chars` | `12000` | 每个会话的近期问答总字符上限 |
| `retrieval_top_k` | `5` | 混合检索每次返回给研究 Agent 的候选片段上限 |
| `score_threshold` | `0.2` | 混合检索最低相关度，范围为 `0`—`1` |
| `embedding_provider_id` | 空 | 指定 AstrBot Embedding Provider；留空时使用第一个可用提供商 |
| `enable_vector_search` | `true` | 是否启用 Chroma 向量检索 |
| `chunk_size` | `1200` | Markdown 分块的目标字符数 |
| `chunk_overlap` | `180` | 超长块滑动切分时的重叠字符数 |
| `retrieval_diagnostics` | `false` | 是否在日志中输出检索与证据诊断摘要 |

轮数上限和字符上限同时生效，任意一个达到上限都会淘汰最早的完整问答轮次。只有成功触发并完成回答的 `/cac` 问答会被写入；普通消息、其他插件消息和失败的模型调用都不会进入上下文。

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

更新文件时保持 UTF-8 编码。插件会根据文件内容自动重建派生索引；SQLite、Chroma 和索引状态保存在插件数据目录，不会改写 `knowledge_pack`。制度、活动时间和报名信息仍应先在知识包中完成人工版本核对。

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

实现参考了 AstrBot 官方插件接口、[Soulter/helloworld](https://github.com/Soulter/helloworld) 模板，并复用了 [Gu-Heping/astrbot_plugin_nju_qa](https://github.com/Gu-Heping/astrbot_plugin_nju_qa) 的两阶段证据 Agent、混合检索和工具化阅读思路；NJU 的远程语雀同步被替换为当前仓库内的本地 Markdown 内容包。

## 许可证

由于 `v0.2.0` 复用并修改了 `astrbot_plugin_nju_qa` 的 AGPL-3.0 代码，插件软件从该版本起按 [GNU AGPL v3](LICENSE) 发布；详细来源说明见 [NOTICE](NOTICE)。`knowledge_pack/` 中的 Markdown 原始材料不因软件许可证而被重新许可，其版权和原始来源条款仍归各自作者所有。
