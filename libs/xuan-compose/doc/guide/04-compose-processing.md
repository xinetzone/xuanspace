# 04 · compose 处理管线：从文件到规范化模型

`xc` 对 compose 文件的处理分四步：**发现 → 插值 → 合并 → 归一化**。
理解这条管线，就能解释 `xc config` 的输出、多文件覆盖的结果与各类报错。

## 用 `xc config` 观察管线

`config` 是唯一「只解析不执行」的命令，是学习管线的窗口：

```bash
xc config                    # 输出发现、插值、合并、归一化后的完整模型
xc config --services         # 仅逐行打印服务名
xc config -q                 # 只解析不打印（CI 中用来校验文件合法性）
xc config --no-normalize     # 跳过归一化，看合并后的原始形态
```

(discovery)=
## 第一步：文件发现

未用 `-f` 指定时，候选名优先级如下（`compose.yaml` 最先）：

```text
compose.yaml → compose.yml
→ compose.override.yaml → compose.override.yml
→ podman-compose.yaml/.yml
→ docker-compose.yml/.yaml
→ docker-compose.override.yml/.yaml
→ container-compose.yml/.yaml
→ container-compose.override.yml/.yaml
```

查找规则（按优先级）：

1. 若设置了 `COMPOSE_FILE` 环境变量或 `.env` 中的同名变量，以
   `COMPOSE_PATH_SEPARATOR`（Windows 为 `;`、类 Unix 为 `:`）分隔的列表为准，
   仅保留实际存在的文件；
2. 否则从**当前工作目录逐级向上**递归（最多 10 层），在**第一个命中候选名的
   目录**收集该目录下全部命中的候选文件（例如同时存在 `compose.yaml` 与
   `compose.override.yaml` 时两者都会被收集，随后在合并阶段叠加），并把工作
   目录切换到该目录；
3. 始终找不到则报错退出，提示用 `-f` 指定。

显式 `-f` 可指定任意文件并多次出现；`-f -` 表示从标准输入读取。
`include:` 段引入的文件按 Compose 规范相对**被包含文件所在目录**解析
其内部相对路径（而非项目根）。

(interpolation-step)=
## 第二步：插值

插值在合并前对每个文件进行，采用 bash 风格变量替换：

| 写法 | 含义 |
|------|------|
| `$VARIABLE` / `${VARIABLE}` | 取值；未设置为空串 |
| `${VARIABLE:-default}` | 未设置**或为空**时取默认值 |
| `${VARIABLE-default}` | 仅未设置时取默认值 |
| `${VARIABLE:?err}` | 未设置或为空时报错（`err` 为消息） |
| `${VARIABLE?err}` | 仅未设置时报错 |
| `${VARIABLE:+alt}` / `${VARIABLE+alt}` | 已设置（或非空）时替代展开 |
| `$$` | 转义为字面量 `$` |

变量来源：进程环境 + 环境文件（默认 `.env`，可用 `--env-file` 追加多个）。
花括号内容不是合法变量名时按上游行为拒绝插值。

```yaml
services:
  web:
    image: nginx:${NGINX_TAG:-stable}
    ports:
      - "${WEB_PORT:-8080}:80"
```

## 第三步：合并

多个 `-f` 文件与 `include:` 文件按 Compose 规范深合并：映射按键合并、
序列按位置/规则叠加。两个特殊 YAML 标签提供强制语义：

- `!override`：以本段整体替换而非合并对应映射/序列；
- `!reset`：清空目标键。

服务还可用 `extends:` 继承另一个服务（或另一个文件中的服务）定义，
解析后再叠加本服务的覆盖；`resolve_extends` 已下沉到规范层，消除了早期的
惰性导入环（结构演进见 [../dev/index.md](../dev/index.md)）。

## 第四步：归一化

归一化把各种等价简写统一为内部规范模型，`--no-normalize` 可关闭。
主要规则（实测自 `normalize.py`）：

- `build: ./dir` 字符串 → `build: {context: ./dir}`；`build.args` 的 map → 列表；
  `additional_contexts` 的 map → `KEY=PATH` 列表；
- `env_file`、`security_opt`、`volumes` 的单字符串值 → 单元素列表
  （`volumes: null` 直接报错）；
- `security_opt` 的 `seccomp:unconfined` / `apparmor:unconfined` → `=` 形式；
- `environment`、`labels` 统一为字典；
- `extends` 字符串 → `{service: 名字}`；
- `depends_on` 的字符串/列表简写 → 字典，且每个依赖补默认条件
  `condition: service_started`；
- `secrets`（含 `build.secrets`）必须是列表，字符串转列表，字典直接报错；
- 相对路径的卷、env_file 在子目录/被包含文件上下文中改按对应基准目录解析。

例如：

```yaml
services:
  web:
    depends_on: [redis]
    environment:
      TZ: Asia/Shanghai
```

归一化后 `depends_on` 等价于：

```yaml
depends_on:
  redis:
    condition: service_started
```

## 处理后的状态落在引擎上

经 CLI 运行时，产物挂在 `ComposeEngine` 上供 handler 与翻译层消费：
`project_name`、`all_services`、`services`、顶层网络/卷/密钥定义等。
在库代码中直接调用 `engine._parse_compose_file()` 即可完成同样四步，
见 [03-library-api.md](03-library-api.md)。

## 下一步

- [../examples/01-web-redis.md](../examples/01-web-redis.md)：带插值与依赖声明的完整文件
- [../reference/commands.md](../reference/commands.md)：`config` 及其他命令选项
