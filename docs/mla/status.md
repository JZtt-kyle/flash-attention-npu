# MLA 当前状态

<!-- mla-loop-next: M18 -->

**M17 非量化已完成；INT8 的 M18–M21 已获用户授权，唯一下一开发步是 M18。** 2026-10-18 用户确认“NZ不需要，迁移INT8就行”，要求拆分任务并准备无头模式；使用入口 `/home/zhangqijia/repo/mla_codex_loop.py`。本次为契约/计划及无头控制收尾，不实施 INT8 数值核，不启动下一开发轮。范围见 [design 3.4](design.md#34-int8-压缩-mla-扩展)，各步路径、检查及完成条件见 [plan](plan.md#int8-后续步骤)。

## 本次改动与验收

公开范围固定为 INT8 内容 Q/共享 KV、BF16 RoPE/输出、FP32 量化 scale，BSND/TND Q + ND PA。外部 NZ 不迁移；固定 CANN INT8 原生仅接受 NZ PA，源对照转换限定在测试适配中，生产读取 ND 并在片上适配布局。尚未接通的量化组合继续拒绝；此处是已授权目标，不是实现声明。

无头外层文件已经转发到版本管理的 `scripts/mla_codex_loop.py`，无须更换用户入口。新增从本页唯一下一步标记读取阶段结束信号，完成后正常退出；未完成轮次和到期文档轮优先处理，失败保留原会话，不跳过验收、不制造空提交。现有模型、登录及 `--approve-for-me` 自动审批沿用。

本次证据根 `/tmp/mla-int8-headless/`：

- `collect.log` 收集 45 项；`tests-final.log` / `tests-final.xml` 实测 45 项通过，16.11 s，其中新增 10 项覆盖阶段结束、恢复优先级、到期文档轮、错误标记、正文引用标记及状态检查。收尾补齐正文引用兼容后重跑所属 45 项，最终全部通过；测试使用真实临时 Git 仓库及模拟 CLI，不调用模型。
- 用户指定入口的 `check-final.json` 显示 `ready`、下一步 M18、pending 为空、CLI 0.163.0 且已登录；`dry-run.log` 预览四轮开发和一次到期文档轮，逐字节确认 state 未改变。旧基线拒绝记录 `check-before.log`，持锁归档记录 `retire.log`；没有伪造旧轮次成功。
- 本次新增/修改的 11 个本地链接及锚点核对通过（`docs-review.log`），`git diff --check` 与六文件 `ci/code_quality.sh --staged` 通过（`quality-final.log`）。格式检查只调整测试函数/调用折行，两次均已核对 AST 未变并重新检查通过。

首次直接 pytest 因全局 seed 夹具导入 torch_npu 而被当前 devlib 占位库缺符号拦截，45 个 setup error/13.06 s 保留在 `tests.log`，单点诊断在 `diagnose.log`。这组工具测试不使用设备，改用下列关闭设备后端自动加载的 CPU 命令后全部通过；对应进程产生的 core 文件已清理，错误日志保留。默认 Python 未安装 pre-commit，复用既有依赖与缓存运行质量入口：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q tests/mla/unit/tools/test_mla_codex_loop.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-precommit-cache ci/code_quality.sh --staged
```

没有修改生产源码、构建配方、CANN 源仓或 CATLASS，也没有探测设备或重新构建扩展。本次 E2E 继承为 0，INT8 设备验证和真实模型开发均未运行。

## 无头运行

```bash
python3 /home/zhangqijia/repo/mla_codex_loop.py --check
python3 /home/zhangqijia/repo/mla_codex_loop.py --dry-run --max-dev-runs 4
python3 /home/zhangqijia/repo/mla_codex_loop.py --max-dev-runs 4
```

最后一条最多完成四个开发提交，到期文档轮不计入上限；M18–M21 每步一次提交，M21 全部验收通过后将本页下一步标记改为 `none`，即使剩余次数未用完也正常退出。中断后重启同一命令恢复同一会话；正常结束但没有提交的阻塞需先处理实际原因，再用 `--retry-blocked`，不能靠清空状态跳过失败。`--dry-run` 只预览调度与命令，不预测未执行的阶段成果。

旧状态中的待办是从 `26145fa` 开始的 M16 失败轮（目录 `20261018-093430-development-7m4fgk2s`），其图采样缺口后来已在 `91a1aaa`/`ebdf3eb` 收尾，归档时 HEAD 已跨两个提交，不能继续恢复该旧基线。本次按脚本持锁归档并保留全部原日志/源码，归档操作不补记成功计数。历史开发/文档计数 97/93 保留，因此下一次成功开发后会自动插入文档核验，之后继续 INT8 下一步。

## 可用基线与环境

非量化基线为 `ebdf3eb`，分支 `feat/mla-cann-migration`。历史完整命令和结果用 `git show ebdf3eb:docs/mla/status.md` 查询，原始证据 `/tmp/mla-m17/{final-tools,final,profile-preflight-retry,performance}/summary.json`。M17 日常 E2E 873、调度专项 402、图专项 233、普通 FA3 504 及普通图 2 项通过；eager 平均比 0.7967466565403458、调度 0.7267397642563352 通过，图 1.484216023946693 失败。用户仅接受 M17 的该已知限制，INT8 新路径仍须满足默认性能门槛。

上一轮环境：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；910B3 设备 7（20 AIC/40 AIV），910C 仅配置兼容、未实机验收。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；CANN checkout HEAD `2060a3db064c3352fec6262faed8a2decaf3a233` 不等于固定源，native/probe 必须用 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的 archive，不能直接编当前 checkout。

已验收扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 为 `14edf068b305b840c5991b5daaef221618e1628ff626722574ac643de317373a`；M18 开始前重新核对环境和实际加载身份。构建/数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts`，原 native 为 `/tmp/mla-m13b/native`，现有配方只有非量化，不能作为 INT8 已准备好的 native 包。

上一轮使用 `/tmp/mla-m16-schedule/run.py` 恢复正确库路径；直接运行 npu-smi 曾因 devlib 占位库缺符号失败。性能的私有 tmpfs 包装器为 `/tmp/mla-m16-schedule/run_private_storage.py`，历史性能在 `unshare -Ur -m` 下执行，输出归档后保留原绝对路径。M18 先核对这些环境/存储条件仍有效，再进行实机和最长 reference 成本检查；失效时修复环境，不伪记通过。

## 唯一下一步

执行 [M18：INT8 数值与 ND 适配](plan.md#m18int8-数值与-nd-适配)。先按计划补用例、核实固定源合法组合和 INT8 native/probe，闭环量化数值、ND 地址与组件性能；公开 INT8 入口留在 M19 接通。M18–M21 已获授权，不因本次公开 dtype/scale 扩展再次询问；出现计划外的新功能或必要验收失败时保留当前工作和交接。
