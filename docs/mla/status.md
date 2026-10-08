# MLA 当前状态

## 当前能力与唯一下一步

- 本次完成 M06b2a：新增 BSND 实际 KV 长度的 v2 metadata 序列化与独立 AICPU 生成，记录为 `401+B` 个 int32，保存真实源调度和逐样本长度快照。M06b2 已按生成（b2a）与设备消费（b2b）拆分；生产构建包含新 AICPU 对象，本步只由独立驱动发射。
- 当前公开能力保持 M04c4：FP16/BF16、512+64 共享 KV、连续 BSND、Hkv=1、无 mask、显式 `num_splits=1`、KV 容量 0..16384。公开 v1 仍为 401 字，要求整容量长度；空容量行为保持原契约。实际长度 Kernel、v2 消费者和公开短长度均未接通。
- 分支 `feat/mla-cann-migration`，基点 `b4bf88c`，接手时工作区干净。唯一下一步为 **M06b2b：BSND 实际长度记录的设备校验消费者**，检查当前长度与快照、省略尾部的零 KV、损坏/过期记录真实 Trap 和拒绝不写输出。范围与完成条件见 [plan](plan.md#m06非分页长度tnd-与-stride)，后续在已确认范围内推进，无需重复确认。

## 本次闭环与证据

- 证据目录 `build/mla-m06b2-metadata/`，最终索引 `acceptance.json`，复核脚本 `audit.py`。`handoff-audit.json` 核对上步 24 个文件哈希、312 项 CPU 通过证据、扩展和只读 CANN 状态；上步原失败已由完整重跑修复，不跳过遗留失败。
- **1,168 项去重 pytest 检查通过，无失败/跳过**。完整 CPU 为 349 项（`cpu.xml`）；补强字段检查后重跑新协议 36 项（`cpu-final.xml`，重复不累加），新增零尾部专项 4 项（`cpu-suffix.xml`），合计 353 项 CPU。覆盖 ASan/UBSan、失败调度/快照清零、缓冲对齐/字数/地址/重叠、静态字数边界及新旧 AICPU 增量依赖；上步 33,736 组长度源差分和既有 dense 最小闭环全部重跑。
- 新协议 CPU 固定源矩阵为 6,484 条记录（4,660 原生非 FD、1,824 FA3 禁止 split 映射）；AICPU 为 3,244 条（2,330 / 914），各含 4 条计算端点之外的零 KV 尾部。映射结果不称为原生输出；同时检查完整调度、几何、资源、workspace 段、任务覆盖/成本和长度快照。源清单、输入与输出摘要保存于 `cpu-final/`、`cpu-suffix/`、`device-verified/` 及其 oracle 目录。
- 910B 21 项全部通过，最终证据 `device-verified.xml`：3,244 条记录与 CPU 及固定源调度一致；覆盖非均匀/等长短序列/全零实际长度、空容量、C1V1/C1V2 合成平台、源端点外的零 KV 尾部、逐位置更新、保护区、事件顺序和 allocator 生命周期。入队阶段禁止 Python D2H 与主机同步；真实平台资源由设备查询。
- 旧 dense 与普通 FA3 共 788 项回归通过（`regressions.xml`），其中普通 FA3 metadata/compile 为 20 项；另有公开生成、隐式生成及复用的短长度真实 Trap 共 6 项通过（`public-short-errors.xml`）。这些结果不代表新 v2 已有设备消费者。
- 双 dtype 各 5 组生成耗时诊断完成，CPU 亲和性 8..15；v2 完整/短长度的 wall 中位数约 244–248 微秒，含 ctypes 发射、双事件和逐调用同步。原始 20 样本/组见 `device-verified/timing-*.json`；不是 attention Kernel 或公开 E2E 验收。本步未修改 Kernel / Block / Tile，实际长度前向精度与 E2E 留给 b3/b4。
- 生产扩展强制重建成功，18 个链接对象、210 个源码/依赖哈希及实际加载路径已核查，见 `production-build-manifest.json`。扩展 SHA256 为 `f77e8a3a17ebac21caaac30e27d507dded962b08b862d51f613e9591ddaf9811`。
- 原始失败均保留：沙箱驱动初始化失败；首次主机探测覆盖 PYTHONPATH 导致 tbe 缺失，保留原路径后成功；首次耗时诊断的 ctypes Args 类型声明不匹配，修复绑定后完整 21 项重跑通过。格式检查曾改写格式，一次误含运行时缓存；缓存已移至忽略证据目录 `runtime-kernel-meta/`。六项适用代码质量钩子已通过，其余三项无对应文件；提交前仅暂存本步 17 个文件执行 `ci/code_quality.sh --staged`，证据为 `quality-staged-*`。差异与 `git diff --check` 已核查；提交一次后停止。

## 环境与后续复现

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。探测到 8 张健康 Ascend910B3；本步生成使用物理设备 5，旧协议/普通 FA3 回归使用设备 6，后续仍须重新核对占用。
- 实际扩展为仓库内 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`。CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f`，工作树干净；只读 CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，接手时已有的 67 条删除/未跟踪记录保留。
- 复现命令和退出码在 `*-command.json` / `*-exit.json`，普通/旧 dense 的确切节点由 `regression-nodes.json` 保存；运行时必须保留 CANN 的 PYTHONPATH。完整 CPU 与新增设备矩阵命令如下；指定 `MLA_BSND_METADATA_CPU_DIR`、`MLA_BSND_METADATA_DEVICE_DIR` 等可保留原始产物。设备命令在主机环境执行，设备选择依据当时占用。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata.py tests/test_flash_attn_npu_v3_mla_bsnd_schedule.py tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu.py
git diff --check
PYTHONPATH="/tmp/mla-m01-precommit:${PYTHONPATH:-}" PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```
