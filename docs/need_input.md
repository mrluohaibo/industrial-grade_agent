我发现你没有将bz_agent/prompts中的提示词模板导入Mongo，是否能先设计一个导入脚本，我先看看可行性，先不执行代码，ultra think

再重新设计一个导入脚本，我先看看可行性，先不执行代码，ultra think


将bz_agent/rag/multi_call_rag_api.py文件中的rrf方法改造成基于bge-reranker-base/large模型的重排序实现，bge-reranker-base/large这模型下载到
model目录下，先制定执行计划，我看看是否合理，先不执行代码变更，ultra think

> pip install modelscope
> modelscope download --model BAAI/bge-reranker-base  --local_dir ./model/bge-reranker-large
> 


运行到 /bz_agent/rag/bge_reranker.py 报错stock_info_app-bge_reranker.py-:86-ERROR-: Failed to encode passages: 'M3Embedder' object has no attribute 'encode_passages'，分析一下原因并提供解决方案