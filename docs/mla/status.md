# MLA 当前状态

实现停在 **M06b2a**（`318fb0a`），公开能力停在 M04c4 加 M05 的整容量 dense 前向与 metadata 复用。本次为文档核验与整理，基于分支 `feat/mla-cann-migration` 的 `318fb0a`，接手时工作区干净；没有修改实现或推进里程碑。

唯一下一步为 **M06b2b：BSND 实际长度记录的设备校验消费者**。范围与完成条件见 [plan 的 M06 表](plan.md#m06非分页长度tnd-与-stride)：校验 v2 头部/字数、快照与当前长度、调度/资源及省略的零 KV 尾部，损坏/过期记录真实 Trap，拒绝不写输出。实际长度 Kernel 在 b3、公开短长度在 b4，后续在已确认范围内推进，无需重复确认。

## 当前支持边界

| 项目 | 代码实际行为 |
| --- | --- |
| 设备与表示 | 生产只开放 **910B C1V2**；FP16/BF16、512+64 共享 KV、Hkv=1。目标仍含 910C，当前 C1V1 合成调度检查不代表 910C 前向已接通 |
| 输入 | 连续 BSND，允许连续视图的非零 storage offset；B/Sq/Sk 可为零，Hq>0，KV 容量 `0..16384`；S1G 已支持多块，不再限于 256 行；字段/资源上界见 [design 6.2](design.md#62-调度字段与边界) |
| 前向入口 | `flash_attn_func` 内部生成 v1；`flash_attn_with_kvcache` 支持只读 dense 的内部生成或显式复用。必须显式 `num_splits=1`，无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`；scale 由本次前向提供 |
| 公开长度与记录 | `get_scheduler_metadata` 的 64/512 分支返回连续 `int32[401]` v1，要求设备 `int32[B]` 长度逐项等于 KV 容量。只读前向接受 None、等于容量的整数或同设备连续 int32 tensor；消费时重新检查当次长度 |
| 空容量与返回 | 空 batch/query 保留空 shape；全局空 KV 为 `O=0、LSE=+inf`，仍校验记录/长度。dense 请求 LSE 返回二元组，kvcache 返回四项，未 split 的两个 accumulator 为空 tensor |
| 独立组件 | `BuildBsndKvLengthSchedule` 与 v2 `int32[401+B]` 序列化/AICPU 生成已验收。v2 已编入扩展，仅由独立驱动发射，**没有设备消费者或前向接入** |
| 尚未开放 | 实际短长度/非均匀长度、TND、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、softcap/其他调度选项、梯度。追加 KV、内置 rotary 等范围外选项同样明确拒绝；完整图 replay 生命周期留在 M16 |

核验入口为 [Python 真实/fake 检查](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 边界](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产生成/发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 与 [设备校验](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_metadata.hpp)。v1 生成遇到非法设备长度时写失败状态并清零 payload；前向消费失败记录或发现当次长度非法才触发 Trap，生成调用本身不立即报该内容错误。

## 最近闭环与证据

**本次文档核验：** 整理三个文件的职责，移除过期的首步边界与阶段流水叙述；按现有代码修正构建单元、launch 参数名、host 调度仍存在及错误上报时机；将协议放回 design、完成条件留在 plan、实际支持与证据留在本页。核验结果保存在 `build/mla-docs-audit/`：`evidence-check.json` 复核历史证据，`doc-checks.json` 核对链接/锚点、源码符号、步骤与文档范围，`quality-staged.log` 保存仓库质量检查。本次只做静态与文档检查，未导入扩展、未重建、未运行 CPU 数值/NPU 精度或性能测试。

57 个本地链接/锚点、18 张表、51 个步骤状态与固定官方公开签名核查通过；`git diff --check` 和三个适用质量钩子通过，其余六个钩子因无对应文件跳过。仅提交这三个文档。

**最近实现闭环仍为 M06b2a。** 证据目录 `build/mla-m06b2-metadata/`，索引 `acceptance.json`；本次重新核对 JUnit、退出码、固定源清单、210 个源码/依赖哈希、18 个链接对象及扩展哈希，一致。归档结果如下，不计作本次重新运行：

| 验收面 | 归档结果 | 证据（相对上述目录） |
| --- | --- | --- |
| CPU 调度/协议/消费者/构建检查 | 353 项去重通过；含 ASan/UBSan、失败清零、缓冲/溢出与新旧增量依赖 | `cpu.xml`、`cpu-final.xml`（36 项重跑不累加）、`cpu-suffix.xml` |
| v2 CPU 固定源记录对照 | 6,484 条：4,660 原生非 FD、1,824 FA3 禁止 split 映射；含 4 条端点外零 KV 尾部 | `cpu-final/`、`cpu-suffix/` 及 oracle 目录 |
| 910B v2 生成 | 21 项通过；3,244 条记录（2,330 原生非 FD、914 映射），含 4 条零尾部；CPU/固定源一致、事件与 allocator 生命周期正确 | `device-verified.xml`、`device-verified/` |
| 旧 dense / 普通 FA3 | 788 项通过，其中普通 FA3 metadata/compile 20 项 | `regressions.xml`、`regression-nodes.json` |
| 公开短长度拒绝 | 显式生成后消费、内部生成后消费、复用前向三种路径，双 dtype 共 6 项真实 Trap 通过 | `public-short-errors.xml` |

合计 **1,168 项去重 pytest 通过，无失败/跳过**；另有 33,736 组实际长度调度源差分保存在 `lengths/`。原生非 FD 与 FA3 无 split 映射分开统计。v2 生成耗时仅是含 ctypes 发射、双事件和逐调用同步的诊断（双 dtype 各五组，wall 中位数约 244–248 微秒），不代表实际长度前向精度或公开 E2E。

最近公开 E2E 闭环来自 M04c4（`85ce13d`）：`build/mla-m04c4-empty/acceptance.json`。本次复核其原始 wall 数据，双 dtype 各 13 组同路径的复用及生成加 R=1/4 前向比值均 `<=1.0`；噪声补测记录 `noise-recheck.json`。它不覆盖 v2/短长度前向；本次及 M06b2a 均未重测公开 E2E。完整 26 组组件矩阵和比较路径规则见 [design 7.1](design.md#71-固定-dense-矩阵与阶段边界)。

上步沙箱驱动初始化、覆盖 PYTHONPATH 丢失 tbe、ctypes Args 绑定及格式检查的原始失败保留在原目录；最终完整重跑与质量检查成功，无遗留失败需越过。历史过程不在本页展开。原始证据为本机 `build/` 下的忽略产物，未随 Git 分发，交接到其他环境时需一并保留。

## 环境与复现

上次 910B 运行环境为 Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0；当时 8 张 Ascend910B3 健康，生成使用设备 5、回归使用设备 6。本次文档步未重新探测运行时或设备占用，后续实机步骤重新确认。

本次核对磁盘扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 与上步生产 manifest 一致，SHA256 为 `f77e8a3a17ebac21caaac30e27d507dded962b08b862d51f613e9591ddaf9811`；这是磁盘产物核验，不是本次加载验证。CATLASS/CANN 固定版本仍与 [design 第 1 节](design.md#1-范围与固定依据) 一致，CATLASS 干净，CANN 原有 67 条删除/未跟踪记录未变且未修改。设备测试前保留 CANN 原 PYTHONPATH，重新核对实际加载扩展。

本次文档核验可复现为：

```bash
python3 build/mla-docs-audit/verify_evidence.py
python3 build/mla-docs-audit/check_docs.py
git diff --check
PYTHONPATH="/tmp/mla-m01-precommit:${PYTHONPATH:-}" PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```

上步最小 CPU 与独立设备矩阵命令如下；原始环境、确切节点、命令和退出码见 `build/mla-m06b2-metadata/*-command.json` / `*-exit.json`。设备 5 是历史选择，重跑前按当时占用调整；用 `MLA_BSND_METADATA_CPU_DIR`、`MLA_BSND_METADATA_DEVICE_DIR` 等指定新的证据目录，保留历史产物。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata.py tests/test_flash_attn_npu_v3_mla_bsnd_schedule.py tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu.py
```

需要构建时仅选择目标后端，header/AICPU/实例集合变化后核查增量依赖，必要时追加 `FLASH_ATTN_FORCE_REBUILD=TRUE`：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位；历史步骤的完整证据说明用 `git log -- docs/mla/status.md` 和对应提交版本查看。
