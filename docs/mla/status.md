# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `959df6d`，接手时工作区干净。M04b2 拆为 Cube、Vector/Kernel、FA3 接入三步；本次完成 M04b2a。
- 新增 `mla/block/mla_cube.hpp`、`mla/tile/mla_copy.hpp`、`mla/tile/mla_mmad.hpp`。从固定 CANN 源迁移 FP16 QK/PV，保留版权和 `LICENSE.cann`；复用 CATLASS `TileMmad`、FP32 `CopyL0CToGm`，CATLASS 子模块未修改。
- 组件范围：连续 BSND、512+64 共享 KV、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1。保留源 N→K→M 顺序、Q/RoPE 跨槽复制、七个 L1 槽、L0 双缓冲、unitFlag、本地事件和跨核事件；调用方负责验证范围及 workspace。
- 独立设备探针以外部 FP16 P 检查 PV，Vector 仅执行源事件协议，没有 softmax/aMLA、scale、O/LSE 或完整 MLA 前向。测试使用 C1V2、独立循环分批，不代表 M04b1 调度已接设备；保留的跨 S2 原子加分支尚未验收。
- 生产 MLA 前向与 MLA metadata 仍明确报未实现；公共入口、生产构建、普通 FA3 源码、已链接对象和实际加载扩展均保持 M03 状态。
- 唯一下一步：M04b2b，按 [design 6](design.md#6-代码结构) 接通 FP16 Vector Block 与 dense Kernel，完成独立 D01/D02、边界、scale、O/LSE、重复执行及同轮 CANN 性能诊断；FA3 接入在 M04b2c。本步提交后停止。

## 本步验证

证据目录：`build/mla-m04b2a-cube/`，最终验收为物理 7 号 910B3。

| 项目 | 结果 |
| --- | --- |
| 上步最小闭环 | PASS：重跑 M04b1 CPU 对照 85 项，无 skip，49,251 组源结果一致；原调度和测试未修改，既有 sanitizer 通过记录已复核，本轮未重跑 sanitizer |
| 910B 组件精度 | PASS：最终新构建 155 项，无 skip；157 组记录、546 个有效双缓冲槽；QK/PV 与固定源 FP32 输出逐元素一致；独立 CPU FP64 计算的最大绝对误差分别为 `3.666e-5` / `2.825e-5`，均满足逐元素 FP32 累加误差界 |
| 边界和复用 | PASS：M 的 1/15/16/17/127/128/129/255/256，KV 的 16/32/128/256/512 两侧，内容/RoPE 四段分别扰动；多 query、非零 storage offset、batch 在 20/40 两侧、反序异形调用、输入及分配边界保护 |
| 构建与来源 | PASS：每次从固定 Git blob 导出原 Cube 及依赖、独立构建两份探针；实际加载路径、源文件/二进制哈希、命令及设备头文件依赖齐全；oracle 不包含迁移代码，迁移探针不依赖原 Cube |
| 代表性组件性能 | 完成诊断：同轮交替 800 次探针调用、400 次 kernel 采样；4 组结果见下表；kernel CSV 与原始 CANN op summary 的名称、设备、时间和时长逐条一致 |
| 文档及提交质量 | PASS：17 处本地链接/锚点、唯一下一步和 metadata E2E 口径一致；差异及 6 项适用质量检查通过；仅本步 9 个文件，最终结果见 `quality-final.log`、`acceptance.json` |
| 完整 MLA、生产扩展构建、普通 FA3 回归 | 未运行：本次未改生产入口、metadata 或既有构建；相关接入及回归留在 M04b2c。M03 的 233 项 MLA 契约和 11 项普通 FA3 证据仍在 `build/mla-interface-guard-validation/` |

最终组件诊断（单位 μs，中位数；shape 为 `B,Sq,Hq,Sk`）：

| shape | 原 Cube kernel | 迁移 Cube kernel | kernel 比值 | 原/迁移探针调用耗时 |
| --- | ---: | ---: | ---: | ---: |
| 2,1,8,257（D01 几何） | 20.690 | 20.511 | 0.9913 | 148.430 / 150.990 |
| 1,7,16,65（D02 几何） | 21.730 | 21.711 | 0.9991 | 180.595 / 172.535 |
| 1,1,128,512 | 46.471 | 46.521 | 1.0011 | 192.290 / 191.590 |
| 2,2,128,511 | 70.451 | 70.652 | 1.0028 | 192.905 / 193.125 |

每例每实现预热 5 次，5 轮交替顺序；每轮墙钟 20 次、profiler 10 次。输入/输出预分配；墙钟包括 Python/ctypes 发射、运行时准备和同步。这里只计 QK 与独立 P×V，没有完整 CANN/FA3 前向或 metadata E2E，不作“不劣化”验收结论。metadata 门槛仍为逐例 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`。

## 环境、失败与限制

| 项目 | 本轮复核 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15，`dav-2201`、`-O2`；最终两组探针均从当前源码重新编译 |
| 芯片与占用 | 7 号 Ascend910B3；Warning `81078603` 为端口 link down，查询时 ECC 计数均为 0；最终性能采样前后仅有本次探针进程。此前 3 号卡采样出现共享进程，仅保留作诊断 |
| 固定来源 | FIA `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净；CANN 源仓已有删除/未跟踪文件保持原样 |
| 实际生产扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `955ecbe4c2b199b2e1131d9c86c38ce23b15ed8704c55cf2e35de37986b84877`，直接加载核实；仍为 M03 产物，不能用于证明新增 Cube 构建 |

早期探针曾编译失败和设备超时：修正源头文件的 host/device 编译隔离、CANN include 路径、探针 Vector 事件顺序，以及迁移时遗漏的 PV FP32 写回后，完成两轮 155 项精度验收；失败日志保留，未计为通过。最终编译/进程退出仍有 CANN 重复 tiling 类型注册警告。源和 CATLASS 的同等写回继续使用原 unitFlag 时序，未通过删除同步规避超时。

Profiler 的框架调用关联解析报 `Failed to get acl to npu flow events` / `fwk_api_db` 失败；探针直接发射 runtime kernel，没有框架算子父节点。最终 400 条 kernel task 数据及原始 CANN CSV 交叉核对通过；不使用框架 flow、CSV 的 block dimension 或硬件计数器作证据。普通 FA3 端到端性能仍待后续接入。

## 证据与复现

- `previous-schedule.xml`、`previous-schedule/`：上步 CPU 重跑及逐组源对照记录；sanitizer 的既有证据在 `build/mla-m04b1-schedule/sanitizers-host/`。
- `environment-initial.json`、`source-repositories.json`、`npu-final-before.json`、`production-unchanged.json`：环境、源仓初始状态、芯片和生产源码/对象/扩展未变化证明。
- `accuracy-final/`、`accuracy-final.log`、`accuracy-final.xml`：固定源原始文件及哈希、两份独立动态库、精确编译/设备依赖命令、构建 manifest、157 组精度记录。
- `performance-final/`、`performance-final.log`：同轮重编译产物、卡占用快照、800 条墙钟记录、400 条 kernel 记录、完整 profiler 原始输出和解析失败日志；`*-command.json`、`*-exit.json` 保存命令、环境与退出状态。
- `smoke*.log`、`debug*.log`、`raw-fixpipe.log` 保存失败；`pv-restored.log`、`accuracy/`、`performance/` 保存修复后及 3 号卡的早期结果，不替代最终记录。
- `verify_acceptance.py`、`acceptance.json`、`quality-final.log`、`checksums.sha256` 保存提交前完整性与质量检查。

从仓库根目录复现（需可访问 910B 驱动；复跑使用新的 artifact 目录，保留旧证据）：

```bash
MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_SCHEDULE_ARTIFACT_DIR=build/mla-m04b2a-cube/recheck-schedule python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_schedule.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_CUBE_ARTIFACT_DIR=build/mla-m04b2a-cube/recheck-accuracy python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_cube.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_CUBE_ARTIFACT_DIR=build/mla-m04b2a-cube/recheck-performance python3 -m tests.mla_cube_profile
```

最终暂存检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04b2a-cube/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
