# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `4f531a3`。**M10c7a BSND PA 显式 split CPU 计划与 v23 安全协议已完成验收。** 唯一下一步为 [M10c7b BSND PA 显式 split 生产接入](plan.md#唯一下一步-m10c7bbsnd-pa-显式-split-生产接入)；本次仅提交 M10c7a，提交后立即停止。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与开放边界

BSND Q + ND PA KV、512+64、S=2..256 的 CPU 计划复用固定源 PA 几何与已有显式分片，覆盖无 mask、右下 causal/window、Q allocation/used、实际 KV 长度、页大小相关 N 基块与原 Q 行地址。静态 M 任务数、持久 partial 行数、scratch 和 FP32 accumulator 分别有独立上界；全零 KV 保留初始化任务，静态空容量无任务但仍验证 accumulator。

v23 使用 `414+7*B` 个 int32，保存逻辑页几何、源调度上界、分配/used 快照、window/S 与宽 M 前缀。独立 CPU/AICore 共用消费者重建预算和任务所有权，逐次检查完整物理 cache 描述与所有有效页，拒绝陈旧长度、交叉版本、损坏记录与预算溢出。页号/物理地址/stride 不绑定记录，合法页表重排可复用；Q-used 不移动原 allocation 行。契约见 [design 6.2.25](design.md#6225-bsnd-pa-显式-split-计划与-v23-安全协议)。

尚未接通 v23 AICPU、设备 partial/merge 或公开 PA split，公开继续明确要求 `num_splits=1`。没有新增设备可调用行为，因此不注册伪 PA split 日常 E2E；后续生产接入独立验收数值、Trap、公开行为和性能。未修改共享数值核、setup、CATLASS 或 CANN 源仓。

## 环境与证据身份

接手工作树干净、分支/HEAD 与交接一致；宿主机无前步遗留 MLA 验收任务，6 号 Ascend910B3 空闲，其他卡任务未改动。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，接手 SHA-256 `c070e52e30d08c60357acc22c8c1df4bd60bc0b965af137fc1d5271175a0787b`，本步重建后 `e0a82d13c44535be0750d38297948fb1022535bfa6fba4a46792e41df67ed0f6`。

证据根 `/tmp/mla-m10c7a/`。构建与数学缓存沿用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；`build/` 仅有 `mla-codex-loop`。前步 E2E 原始文件、源码、构建输入/依赖/产物和加载库共 7,322 个文件核验无差异，见 `takeover.json`。本步新增头文件和驱动改变保守构建/证据身份，累计 E2E 全部实际执行，继承 0；未累计历史单元或性能。

## 本次通过

最终各组均无失败或跳过，节点与各自 collect-only 清单逐项一致；普通和 ASan/UBSan 同为 337 项集合。工具组与普通单元后补的测试参数修正无依赖关系，其原始证据及源码也已重新核验。CPU 工具、固定 CANN/FA3、普通/sanitizer 二进制与原始结果共 1,133 个唯一文件核验通过，见 `evidence-cpu-final.json`；最终 `evidence-final.json` 核验源码、原始结果、构建和加载库共 7,474 个唯一文件，差异为 0。

| 组 | 通过 | 耗时 | 证据目录（相对证据根） |
| --- | --- | --- | --- |
| `split-paged-bsnd-protocol-tools` | 14/14 | 8.7 秒 | `cpu-frozen/attempt-1792025111925117470/` |
| `unit-split_paged_bsnd_protocol` | 337/337 | 42.8 秒 | `cpu-final/attempt-1792025258902594950/` |
| `split-paged-bsnd-protocol-sanitize` | 337/337 | 140.6 秒 | `sanitize-final/attempt-1792025466349788100/` |
| `e2e` | 201/201 | 32.0 秒 | `e2e-verified/attempt-1792025659321291790/` |

固定源对照重新编译并执行 PA tiling、mask 范围/workspace 与 FA3 分片/accumulator 片段。独立枚举行、逻辑页段、任务/merge 所有权及原 allocation 地址；覆盖 P=16/112/512/528/1024、三 mask、双 dtype 协议、尾页/跨页、空片/全空/used、多波、M=128/249/256、宽任务号、损坏端点/资源/页描述符及失败原子性。最大实际 CPU 成本输入 B=65538、Qcap=65536、P=C=16、H=1、S=256，实际查询超过 uint32 的任务和原 Q 行地址；只分配 O(B) 计划/记录，不申请逻辑 workspace 或枚举全部任务。

生产扩展及探针完整构建约 522.1 秒；新运行内容核验复用 1.2 秒、实际加载检查 20.0 秒通过。790 个构建输入、6,401 个依赖和 12 个产物与当前内容一致。日常 201 项 E2E 清单与前步完全一致，数学缓存命中 348 次，新计算 0 次，待测算子全部实际执行。236 个既有 suite 仅 `unit-tools` 因自动发现新增本模块工具文件而扩展，其余命令不变；新增本模块四个入口，见 `selection-audit.json`。没有用旧无 mask 数值结果冒充 PA split 能力验收。

## 复现与开发记录

以下命令均使用统一入口，先 `--dry-run` / `--collect-only`。设备与 sanitizer 运行使用宿主机非登录 shell；新运行使用新的 output 目录：

```bash
python3 -m tests.mla --suite split-paged-bsnd-protocol-tools --output /tmp/mla-m10c7a/repro-tools
python3 -m tests.mla --suite split-paged-bsnd-protocol-check --output /tmp/mla-m10c7a/repro-check
python3 -m tests.mla --unit split_paged_bsnd_protocol --output /tmp/mla-m10c7a/repro-unit
python3 -m tests.mla --suite split-paged-bsnd-protocol-sanitize --output /tmp/mla-m10c7a/repro-sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c7a/repro-e2e
```

开发代表/最大成本 9 项通过（39.7 秒），开发模块 280 项通过（42.4 秒）；复核补入独立 M 因子、workspace/本地内存字段损坏和精确溢出。首次冻结模块 334 项通过后，M=128 对照失败：迁移侧 26 核，源辅助函数漏传核数而默认 20 核；修正为两侧 26 核后最终 337 项通过，生产实现未改。该次 sanitizer 未运行。随后沙箱 sanitizer 的 337 项断言通过，但退出时 LeakSanitizer 报 ptrace 限制，整组仍记失败；保持全部选项在宿主机重新运行，同一集合及退出/泄漏检查通过。

初次沙箱 E2E 收集因 NPU 不可见失败，宿主机重新收集通过。`e2e-frozen/` 的构建子命令成功，但测试参数修正改变了驱动整树快照，被标记 `Source changed during preparation`，未执行 E2E；构建输入本身未变，`e2e-verified/` 核验内容后复用并实际完成加载和 E2E。一次提前启动被设备锁拒绝，见 `e2e-final.log`，未绕过锁或运行测试。以上失败/未运行记录保留，未改写为通过。

默认 PATH 无 pre-commit，临时 cache 缺 actionlint；使用既有 `/tmp/mla-m01-precommit` Python 包与完整 `/root/.cache/pre-commit` 完成质量检查，没有绕过钩子。最终差异、相关链接/表格与质量检查见 `doc-links-final.json`、`quality-final.log`；验收进程全部退出且 6 号卡空闲，见 `processes-final.json`、`npu-final.log`。未运行 PA split 设备数值/Trap/公开性能、其他历史单元、ordinary 专项、full-matrix、完整图重放或 910C 实机，不计通过或继承。本次提交后不继续下一步。
