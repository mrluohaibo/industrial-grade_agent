import json

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage, AIMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import List, Optional, Iterator, Any, Dict, Literal, Union, Sequence, Type, Callable

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch
from langchain_core.utils.function_calling import convert_to_openai_function, convert_to_openai_tool


class StreamingLocalQwenChat(BaseChatModel):
    """支持流式的本地 Qwen 聊天模型（LangChain ChatModel 接口）"""

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    def __init__(self, model_path: str, **kwargs):
        super().__init__()
        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto"
        )

        # 配置生成参数
        self.max_new_tokens = kwargs.get("max_new_tokens", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非流式生成"""
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成 —— 核心方法"""
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": True,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # 在子线程中启动生成
        thread = Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()

        # 流式 yield Chunk
        for new_text in streamer:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=new_text))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(new_text)

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""

        # 如果绑定了工具，插入工具说明
        if hasattr(self, '_tools') and self._tools:
            tools_desc = "\n".join(
                f"- {func['name']}: {func['description']} (参数: {json.dumps(func['parameters'])})"
                for func in self._tools
            )
            system_with_tools = (
                "你可以使用以下工具来回答问题：\n"
                f"{tools_desc}\n"
                "如果你需要使用工具，请严格按以下 JSON 格式回复（不要包含其他内容）：\n"
                '{"tool_name": "xxx", "arguments": {"arg1": "value1"}}\n'
                "如果不需要工具，请直接回答。"
            )
            # 插入到第一条消息前（或合并到 system message）
            new_messages = []
            inserted = False
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    new_messages.append(SystemMessage(content=system_with_tools + "\n" + msg.content))
                    inserted = True
                else:
                    new_messages.append(msg)
            if not inserted:
                new_messages = [SystemMessage(content=system_with_tools)] + list(messages)
            messages = new_messages

        # 继续构建 Qwen prompt
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif isinstance(msg, AIMessage):
                prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    @property
    def _llm_type(self) -> str:
        return "streaming-local-qwen-chat"



    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:

                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``: automatically selects a tool (including no tool).
                - ``"none"``: does not call a tool.
                - ``"any"`` or ``"required"`` or ``True``: force at least one tool to be called.
                - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                - ``False`` or ``None``: no effect, default OpenAI behavior.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. If True, the input schema will be
                validated according to
                https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
                If False, input schema will not be validated and model output will not
                be validated.
                If None, ``strict`` argument will not be passed to the model.
            parallel_tool_calls: Set to ``False`` to disable parallel tool use.
                Defaults to ``None`` (no specification, which allows parallel tool use).
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_openai.chat_models.base.ChatOpenAI.bind`.

        .. versionchanged:: 0.1.21

            Support for ``strict`` argument added.

        """  # noqa: E501

        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)