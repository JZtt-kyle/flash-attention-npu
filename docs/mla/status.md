# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`。M03 接口边界的构建与回归验收已完成。
- 生产入口识别 512+64 压缩 MLA 与 192/128 prefill，校验 rank、shape、dtype、device、头数和 stride；MLA 前向与 MLA metadata 明确报未实现。
- Python 公开入口、自定义算子、fake 和 C++ 入口采用一致的拒绝契约。普通 FA3 使用既有计算与 metadata 路径。
- 独立数学 reference、CANN 输入映射、固定源码基线包、分派溯源和 profiler 工具可用。
- 实现原则：保持固定 CANN 的计算顺序、精度、切块分核、缓冲和同步行为，按 CATLASS Kernel / Block / Tile 组织代码；必要 FA3 接口差异单独映射。
- 唯一下一步：M04a，核对源文件/函数、调度字段、资源布局和事件依赖，确定首个 dense 有界实现范围及独立验收子步骤。性能验收使用 metadata 路径 E2E 与同轮 CANN E2E 的逐例比较。本次提交成功后停止。

## 本步验证

证据目录：`build/mla-interface-guard-validation/`。

| 项目 | 结果 |
| --- | --- |
| 源码边界 | PASS：MLA 目录仅含校验 header；生成器输出 8 个普通 FA3 编译单元；reference 和 CANN 基线可用 |
| 强制构建 | PASS：v3/910，9 个 C++ 编译单元和 1 个 AICPU 对象；实际加载扩展与链接产物哈希一致，普通 FA3 的 8 个 kernel 对象一致性检查通过 |
| CPU reference / provenance / profile | PASS：88 项，无 skip |
| MLA 拒绝契约 | PASS：233 项，无 skip；覆盖两种表示、dense/TND/PA、真实/fake/C++/公开入口、非法参数及未实现选项 |
| 普通 FA3 | PASS：11 项，无 skip；精度、metadata、dense/varlen backward、kvcache/FlashDecode 及编译图回归 |
| 设备记录 | 完整：物理 2 号共享 Warning 卡，21 个快照；前后告警为 81078603，ECC 八项计数均为 0 |
| 文档及质量 | PASS：文档链接、当前状态、差异检查和 6 项适用质量检查；仅暂存本步 20 个文件 |

上述实机结果为功能回归；MLA 数值与 metadata E2E 在对应实现步骤验收。

## 环境与设备

| 项目 | 实际记录 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN | `/usr/local/Ascend/cann-9.0.0-beta.2` |
| 固定 FIA | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；基线包 `build/mla-m02b-validation/opp-dense/` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净 |
| 实际加载扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` |
| 扩展 SHA256 | `955ecbe4c2b199b2e1131d9c86c38ce23b15ed8704c55cf2e35de37986b84877` |
| 芯片 | Ascend910B3，物理 2 号，进程内 npu:0 |

设备为共享 Warning 条件，告警表示端口链路状态变化。开始时 AICore 0%，后台 PID 3724055 占 140 MB；监控文件保存测试期间的实际占用。该记录仅说明本轮功能验证条件，不作为健康、独占设备或性能验收结论。

CANN 源仓只读，已有工作区改动保留；来源核对使用固定 commit。CANN 基线精度与 profiler 数据分别见 `build/mla-m02b-validation/` 和 `build/mla-m02b2-validation/`。

## 证据与复现

- `build-command.json`、`build.log`、`build-provenance.json`：编译参数、10 个链接对象、源码哈希及实际加载扩展。
- `cpu-command.json`、`cpu.xml`、`npu-command.json`、`npu.xml`：88 项 CPU、233 项契约和 11 项普通 FA3 的精确参数与结果。
- `device-readiness.json`、`npu-device-monitor.json`：设备健康、告警、ECC 和共享占用。
- `acceptance-integrity.json`、`source-repositories.json`、`document-check.json`、`quality-final.log`、`checksums.sha256`：来源、结果及提交完整性。

最小命令如下；实机运行前重新检查设备和扩展，保持 CANN PYTHONPATH，使用新日志标签与 JUnit 路径保存本轮结果：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_provenance.py
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_contract.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```

普通 FA3 的精确节点保存在 `npu-command.json`。提交前完整性检查使用 `python3 build/mla-interface-guard-validation/verify_acceptance.py --staged`，该命令对应本步暂存状态。
