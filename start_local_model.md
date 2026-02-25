# pip install vllm


# 启动服务（假设模型已下载到 ～/.cache/huggingface/hub/...）
# linux使用
python -m vllm.entrypoints.openai.api_server --model F:/python_pro/tick_info/model/Qwen3-0.6B --dtype auto --port 8000

# windows环境，仅使用于开发
ollama list
ollama run qwen3