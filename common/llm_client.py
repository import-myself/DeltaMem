"""
LLM Client Module

独立的 LLM 调用模块，支持多种 LLM 客户端：
- OpenAI API (gpt-4, gpt-3.5-turbo 等)
- 自定义 LLM 客户端

提供统一的调用接口，屏蔽底层实现差异。
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    """
    统一的 LLM 客户端接口
    
    支持两种使用方式：
    1. OpenAI 风格：使用 OpenAI API
    2. 自定义风格：实现 generate() 方法
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        backend: str = "openai"
    ):
        """
        初始化 LLM 客户端
        
        Args:
            model_name: 模型名称（如 gpt-4, gpt-3.5-turbo）
            api_key: API 密钥（如不提供，从环境变量读取）
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            backend: 后端类型 ("openai" 或 "custom")
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = backend
        
        if backend == "openai":
            self._init_openai_client(api_key, base_url=base_url)
        else:
            self.client = None
            logger.info(f"Using custom backend for model: {model_name}")
    
    def _init_openai_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            
            api_key = api_key or os.getenv("API_KEY")
            base_url = base_url or os.getenv("BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            if not base_url:
                raise ValueError("OPENAI_API_BASE_URL not found in environment variables")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
            
        except ImportError:
            logger.error("OpenAI library not installed. Install it with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        调用 LLM 进行对话
        
        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            temperature: 生成温度（覆盖默认值）
            max_tokens: 最大生成 token 数（覆盖默认值）
        
        Returns:
            LLM 生成的回复文本
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.backend == "openai":
            return self._chat_openai(messages, temp, max_tok)
        else:
            # 自定义后端需要实现 generate() 方法
            return self._chat_custom(messages, temp, max_tok)
    
    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """OpenAI API 调用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking":False
                    }
                }
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _chat_custom(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """自定义后端调用（需要子类实现）"""
        if hasattr(self, 'generate'):
            # 调用自定义的 generate 方法
            return self.generate(messages)
        else:
            raise NotImplementedError(
                "Custom backend requires implementing generate() method"
            )


def create_llm_client(
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    backend: str = "openai"
) -> LLMClient:
    """
    创建 LLM 客户端的工厂函数
    
    Args:
        model_name: 模型名称
        api_key: API 密钥
        temperature: 生成温度
        max_tokens: 最大生成 token 数
        backend: 后端类型
    
    Returns:
        LLMClient 实例
    
    Example:
        >>> # OpenAI 后端
        >>> client = create_llm_client("gpt-4")
        >>> response = client.chat([{"role": "user", "content": "Hello"}])
        
        >>> # 自定义后端
        >>> client = create_llm_client("my-model", backend="custom")
        >>> # 需要为 client 添加 generate() 方法
    """
    return LLMClient(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        backend=backend
    )


# 兼容旧版本的接口（保持向后兼容）
def create_openai_client(
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    """
    创建 OpenAI 客户端（兼容接口）
    
    返回一个类 OpenAI 对象，可直接用于 agent
    """
    client = create_llm_client(model_name, api_key, base_url,backend="openai")
    
    # 返回一个包装对象，兼容旧的使用方式
    class CompatibleClient:
        def __init__(self, llm_client: LLMClient):
            self._llm = llm_client
            self.model_name = llm_client.model_name
            
        @property
        def chat(self):
            """模拟 OpenAI.chat 属性"""
            return self
        
        @property
        def completions(self):
            """模拟 OpenAI.chat.completions 属性"""
            return self
        
        def create(self, model: str, messages: List[Dict], **kwargs):
            """模拟 OpenAI.chat.completions.create() 方法"""
            # 创建一个类似 OpenAI response 的对象
            class Response:
                def __init__(self, content: str):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': content
                        })()
                    })()]
            
            result = self._llm.chat(messages, **kwargs)
            return Response(result)
        
        def generate(self, messages: List[Dict]) -> str:
            """提供 generate() 方法供自定义后端使用"""
            return self._llm.chat(messages)
    
    return CompatibleClient(client)
