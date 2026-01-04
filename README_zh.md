# nanochat（简版）

nanochat 是一个在单机 8×H100 节点上即可完成从分词、预训练、微调到部署对话 UI 的开源 LLM 项目，代码干净、依赖精简，方便你按需调整和实验。

- 快速体验：在 GPU 服务器上运行 `bash speedrun.sh`，约 4 小时得到可对话的 d32 模型。
- 部署聊天：激活虚拟环境后执行 `python -m scripts.chat_web`，按提示访问网页与模型对话。
- 自定义能力：通过修改 `scripts` 里的训练脚本或参考 `dev/gen_synthetic_data.py` 生成个性化数据。
- 资源提示：显存不足可调低 `--device_batch_size`，单卡或 CPU/MPS 也能以更小配置运行。
