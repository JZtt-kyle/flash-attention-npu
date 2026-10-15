# MLA 当前状态

**本轮为专门文档修正，功能基线仍是 M11a。** 分支 `feat/mla-cann-migration`，接手提交 `fd7e133`，接手时工作区干净。本轮只修改 design/plan/status，合并重复协议与实施历史，修正组件/公开能力、性能适用范围及交接路径；不推进功能、不修改无头循环计数。唯一下一步仍是 [M11b：完整 BSND prefill 接入](plan.md#唯一下一步-m11b)，本轮提交后停止。

## 能力与未开放边界

| 范围 | 当前状态 |
| --- | --- |
| 512+64、Hkv=1、FP16/BF16 | BSND、TND、TND Q + BSND KV、BSND PA、TND PA 五布局的单/显式/自动 split 与无 mask/右下 causal/window 已接通；M10 阶段已收尾 |
| 192/128、独立 K/V、GQA | M11a 通用数值组件已完成三 mask、direct/partial/128 维 merge；**公开前向及 metadata 仍明确拒绝**，没有 prefill 日常公开 E2E 或公开性能通过声明 |
| softcap、非默认调度 | 尚未迁移；保持 `softcap=0`、`pack_gqa=None`、`sm_margin=0`，其他拒绝项见 [design 2.2](design.md#22-后端选项与返回值) |
| 图与芯片 | 已有代表 fake/compile 和流生命周期检查；完整 graph replay 未验收。目标为 910B/C，实现验收使用 910B，未作 910C 实机验收 |

M11a 的组件参数不是新 metadata 协议，组件资源界限和固定源映射见 [design 6.5.1](design.md#651-192128-通用非量化数值组件)。归并夹具以 KV head 映射虚拟 batch，公开 BSND 头布局、完整生命周期和性能仍须 M11b 验收。现有公开性能过滤器只接受 decode template 5；通用 prefill 分派的适用规则也属于 M11b 接入范围。

## 最近开发闭环与证据

最近开发闭环为 `fd7e133`（M11a）。证据根 `/tmp/mla-m11a/`，最终组件组在 `acceptance-final/summary.json`，回归在 `regression/summary.json`。各组的实际目录由 summary 的 `attempt` / `groups[*].directory` 指定，不能省略 `attempt-*` 层级查找日志或 profile。以下均为**上次实测**，本轮没有重跑：

| 检查 | 上次结果（均 0 failed/skipped） | 上次用时 |
| --- | --- | --- |
| prefill 工具/缓存/失败传播 | 22 passed | 7.8 s |
| prefill 模块单元 | 56 passed：27 数值配置 × 双 dtype、2 发射失败原子性 | 253.7 s，含组件重建 |
| 性能廉价夹具 | 2 passed | 26.3 s |
| 组件性能 | 2 passed，12 attention + 6 merge 配置及迁移适配成本 | 67.0 s |
| 累计日常 MLA E2E | 314 passed | 68.0 s |
| 普通 FA3 回归 | 516 passed | 27.5 s |

M11a 最终运行未使用 `--resume`、`--max-cases` 或完整矩阵扩展，继承结果为 0。54 份数值报告的 1860 个双数学对照状态全部 passed，无源精度限制豁免；193..256 片超出原生 FD 归并缓冲域，具名 256 片用例按数学与空行契约验收，不计为原生逐元素覆盖。原始检查汇总为 `final-check.json`。

组件 Kernel 的迁移/固定源配置均值：attention FP16/BF16 为 0.9998/0.9827，merge 为 1.8487/1.8526；相应退化数为 4/6、2/6、3/3、3/3。merge 退化保留，不解释为公开性能已通过或失败。完整 Kernel/wall 样本位于最终 `prefill-profile` 组的 `probes/<dtype>/profile/`；`adaptation-wall.json` 单列迁移行适配与 accumulator 成本。计时保持 5 次预热、5×20 wall、5×10 Kernel、预分配及同片段，源输出初始化保留。

代表与最大成本预检查在 `check-platform/`，共 10 项通过。成本输入 `B=1,Sq=257,Hq=8,Sk=4097,Hkv=2,M=512,N=512,S=8` 在最终组件组缓存后 FP16/BF16 用例耗时为 1.95/2.56 s，报告的进程最大 RSS 约 2.31 GiB；这些是整个用例及进程的测量，不能当作独立 reference 耗时/峰值。开发期 `check-expanded/` 曾把容量 26 当作发射核数导致源同步超时，改为实际物理核数分波后最终组通过；早期失败日志不计为通过，详细修正从 `git show fd7e133:docs/mla/status.md` 查询。

M10 阶段证据及性能结论从 `git show feb189b:docs/mla/status.md` 查询。本轮不把旧阶段源码/扩展身份当作当前全矩阵验收；M11a 改动共享归并后已实跑上述累计 E2E 和普通 FA3，未重跑历史公开专项、性能或阶段矩阵。尚未开放功能没有补记通过。

## 本轮文档核对

本轮证据在 `/tmp/mla-doc-review-20261015/`。`evidence-review.json` 复用统一驱动的 `reusable`、`suite_sources`、`validate_result`、`validate_causal_builds` 和 `matching_files`，只读核对最终六组原始哈希、收集/实测清单、源码依赖、组件/生产构建及保存的加载身份，8.2 s 全部通过；双 dtype 的原生分派日志/二进制仍匹配，六次请求均为通用 template 3。

这些是文件与历史证据核对，不是本次设备通过或 `--resume` 继承；本轮新增设备测试和继承执行结果均为 0。没有探测芯片、加载 torch/扩展、重建或运行设备矩阵。下次开发仍需核对运行环境与实际加载身份，只有符合 [继承规则](design.md#73-增量验收与证据继承) 的 E2E 正确性可复用，历史单元和性能不跳过执行。

固定 CANN 通过只读 Git 对象核对 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，CATLASS gitlink/checkout 为 design 的固定版本；FA3 本地副本位于 `/tmp/mla-fa3-reference/d15f1531a460ba456f41b01a774f33ab2db8febf/`，窗口与 split 源文件按现有 oracle 的 SHA256 校验。GitHub 固定源码外链在线获取返回 cache miss，未记为在线通过；保留固定链接，接口核对使用本地版本副本。

一致性核对覆盖公开 real/fake 拒绝与分派、v1–v26 编码、split 发射/预算、数值组件、固定来源和验收选择。350 个本地链接/锚点、37 个表格及 13 条 `--list`/dry-run 命令通过，记录为 `document-links.json`、`command-review.json`；原生包构建命令仅检查模块路径，没有执行。源码核对记录为 `source-review.json`。

质量检查使用已有 pre-commit 4.6.2：`PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-precommit-cache ci/code_quality.sh --staged`，三份文档适用检查通过，其余代码类钩子按文件类型跳过；`git diff --check` 通过。收尾更新后以同一命令复核，最终日志为 `quality-final.log`。本轮没有新增设备通过结果，也没有未解决的本地检查失败；外链在线核验限制如上。

## 环境记录与复现

以下来自 M11a 保存的 runtime/build manifest，**本轮未重新探测**：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；卡 6 为 Ascend910B3、20 AIC。保存的实际加载扩展为 `/home/zhangqijia/repo/flash-attention-npu/flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `23f3468d4c302c579a7eb122520fa2ba76181f4f67b5dfefe7c372fb8761abfb`，本轮仅确认文件哈希仍相同。

`MLA_WORK_DIR` 默认 `/tmp/flash-attention-npu-mla-46928a793876`：生产/reference 在 `mla-validation-artifacts/`，prefill 组件缓存在 `prefill-probes/`。native 包为 `/tmp/mla-m10c3b/native/opp/vendors/mla_ca18aee_transformer`；M11a 使用已有 103/104 配方，实际分派见证据根的 `native-fp16-dispatch.json` / `native-bf16-dispatch.json`。这些临时目录被清理后不可继续继承。

以下为开发复现命令，本轮只核对选择和 dry-run；运行前先以相同参数 `--dry-run`、再 `--collect-only`。需查询芯片的收集和实测在宿主执行，纯文档轮不运行：

```bash
python3 -m tests.mla --suite prefill-tools --output /tmp/mla-m11a/repro-tools
python3 -m tests.mla --suite prefill-check --device 6 --output /tmp/mla-m11a/repro-check
python3 -m tests.mla --unit prefill_attention --device 6 --output /tmp/mla-m11a/repro-unit
python3 -m tests.mla --suite prefill-profile-check prefill-profile \
  --device 6 --output /tmp/mla-m11a/repro-profile
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla \
  --suite e2e ordinary --device 6 --output /tmp/mla-m11a/repro-regression
```

日常入口仍为 `python3 -m tests.mla --device 6`；原生通用分派预检查用法见 `python3 -m tests.mla.support.mla_prefill_native --help`。历史步骤命令、失败修正和原始证据由 `git log -- docs/mla/status.md` 查询；当前文档闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
