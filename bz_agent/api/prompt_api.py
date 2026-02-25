"""
Prompt 管理 API

为 Web 界面提供 Prompt 模板 CRUD 操作的 FastAPI 端点
"""
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from bz_agent.storage import prompt_store
from utils.logger_config import logger


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


# ============ Request/Response Models ============

class PromptCreate(BaseModel):
    """创建 Prompt 的请求模型"""
    prompt_name: str = Field(..., description="Prompt 名称（不带扩展名）")
    template: str = Field(..., description="Prompt 模板内容")
    description: str = Field("", description="Prompt 描述")
    created_by: str = Field("system", description="创建者")


class PromptUpdate(BaseModel):
    """更新 Prompt 的请求模型"""
    template: str = Field(..., description="Prompt 模板内容")
    description: Optional[str] = Field(None, description="Prompt 描述（可选）")


class PromptResponse(BaseModel):
    """Prompt 响应模型"""
    _id: str = Field(..., description="文档 ID")
    prompt_name: str = Field(..., description="Prompt 名称")
    template: str = Field(..., description="模板内容")
    description: str = Field(..., description="描述")
    active: bool = Field(..., description="是否激活")
    version: int = Field(..., description="版本号")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")
    created_by: str = Field(..., description="创建者")
    tags: List[str] = Field(..., description="标签")


class ImportResponse(BaseModel):
    """导入响应模型"""
    imported_count: int = Field(..., description="导入的 Prompt 数量")
    message: str = Field(..., description="响应消息")


# ============ API 端点 ============

@router.get("/{prompt_name}", response_model=PromptResponse)
async def get_prompt(prompt_name: str, version: Optional[int] = None):
    """
    获取指定名称的 Prompt 模板

    Args:
        prompt_name: Prompt 名称
        version: 可选版本号，不指定则获取最新激活版本

    Returns:
        Prompt 模板信息
    """
    logger.info(f"API: Get prompt '{prompt_name}' version {version}")

    # 获取 prompt
    template = prompt_store.get_prompt(prompt_name, version)

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    # 获取完整信息
    prompt_list = prompt_store.list_prompts(
        prompt_name=prompt_name,
        limit=1
    )

    if not prompt_list:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    prompt_doc = prompt_list[0]

    return PromptResponse(
        _id=prompt_doc.get("_id", ""),
        prompt_name=prompt_doc.get("prompt_name", ""),
        template=prompt_doc.get("template", ""),
        description=prompt_doc.get("description", ""),
        active=prompt_doc.get("active", False),
        version=prompt_doc.get("version", 1),
        created_at=prompt_doc.get("created_at", "").isoformat() if prompt_doc.get("created_at") else "",
        updated_at=prompt_doc.get("updated_at", "").isoformat() if prompt_doc.get("updated_at") else "",
        created_by=prompt_doc.get("created_by", ""),
        tags=prompt_doc.get("tags", [])
    )


@router.post("/", response_model=PromptResponse)
async def create_prompt(data: PromptCreate):
    """
    创建新的 Prompt 模板

    Args:
        data: Prompt 创建请求数据

    Returns:
        创建的 Prompt 信息
    """
    logger.info(f"API: Create prompt '{data.prompt_name}'")

    # 保存 prompt
    doc_id = prompt_store.save_prompt(
        prompt_name=data.prompt_name,
        template=data.template,
        description=data.description,
        created_by=data.created_by
    )

    # 获取保存的完整信息
    prompt_list = prompt_store.list_prompts(
        prompt_name=data.prompt_name,
        limit=1
    )

    if not prompt_list:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create prompt"
        )

    prompt_doc = prompt_list[0]

    return PromptResponse(
        _id=prompt_doc.get("_id", ""),
        prompt_name=prompt_doc.get("prompt_name", ""),
        template=prompt_doc.get("template", ""),
        description=prompt_doc.get("description", ""),
        active=prompt_doc.get("active", False),
        version=prompt_doc.get("version", 1),
        created_at=prompt_doc.get("created_at", "").isoformat() if prompt_doc.get("created_at") else "",
        updated_at=prompt_doc.get("updated_at", "").isoformat() if prompt_doc.get("updated_at") else "",
        created_by=prompt_doc.get("created_by", ""),
        tags=prompt_doc.get("tags", [])
    )


@router.put("/{prompt_name}", response_model=PromptResponse)
async def update_prompt(prompt_name: str, data: PromptUpdate):
    """
    更新现有的 Prompt 模板（创建新版本）

    Args:
        prompt_name: Prompt 名称
        data: Prompt 更新请求数据

    Returns:
        更新后的 Prompt 信息
    """
    logger.info(f"API: Update prompt '{prompt_name}'")

    # 更新 prompt
    success = prompt_store.update_prompt(
        prompt_name=prompt_name,
        template=data.template,
        description=data.description
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    # 获取更新后的完整信息
    prompt_list = prompt_store.list_prompts(
        prompt_name=prompt_name,
        limit=1
    )

    if not prompt_list:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    prompt_doc = prompt_list[0]

    return PromptResponse(
        _id=prompt_doc.get("_id", ""),
        prompt_name=prompt_doc.get("prompt_name", ""),
        template=prompt_doc.get("template", ""),
        description=prompt_doc.get("description", ""),
        active=prompt_doc.get("active", False),
        version=prompt_doc.get("version", 1),
        created_at=prompt_doc.get("created_at", "").isoformat() if prompt_doc.get("created_at") else "",
        updated_at=prompt_doc.get("updated_at", "").isoformat() if prompt_doc.get("updated_at") else "",
        created_by=prompt_doc.get("created_by", ""),
        tags=prompt_doc.get("tags", [])
    )


@router.delete("/{prompt_name}")
async def delete_prompt(prompt_name: str):
    """
    删除指定名称的所有 Prompt 版本

    Args:
        prompt_name: Prompt 名称

    Returns:
        删除结果消息
    """
    logger.info(f"API: Delete prompt '{prompt_name}'")

    # 删除 prompt
    count = prompt_store.delete_prompt(prompt_name)

    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    return {
        "message": f"Deleted {count} version(s) of prompt '{prompt_name}'",
        "count": count
    }


@router.get("/", response_model=List[PromptResponse])
async def list_prompts(
    active_only: bool = False,
    tag: Optional[str] = None,
    prompt_name: Optional[str] = None,
    limit: int = 100
):
    """
    列出所有 Prompt 模板

    Args:
        active_only: 是否只返回激活的 prompt
        tag: 按标签过滤
        prompt_name: 按 prompt 名称过滤
        limit: 返回数量限制

    Returns:
        Prompt 模板列表
    """
    logger.info(f"API: List prompts (active_only={active_only}, tag={tag}, name={prompt_name})")

    prompt_list = prompt_store.list_prompts(
        active_only=active_only,
        tag=tag,
        prompt_name=prompt_name,
        limit=limit
    )

    # 转换为响应模型
    responses = [
        PromptResponse(
            _id=p.get("_id", ""),
            prompt_name=p.get("prompt_name", ""),
            template=p.get("template", ""),
            description=p.get("description", ""),
            active=p.get("active", False),
            version=p.get("version", 1),
            created_at=p.get("created_at", "").isoformat() if p.get("created_at") else "",
            updated_at=p.get("updated_at", "").isoformat() if p.get("updated_at") else "",
            created_by=p.get("created_by", ""),
            tags=p.get("tags", [])
        )
        for p in prompt_list
    ]

    return responses


@router.post("/import", response_model=ImportResponse)
async def import_prompts():
    """
    从文件批量导入 Prompt 模板到数据库

    Returns:
        导入结果信息
    """
    logger.info("API: Import prompts from files")

    # 导入 prompts
    count = prompt_store.import_all_from_directory("bz_agent/prompts")

    return ImportResponse(
        imported_count=count,
        message=f"Successfully imported {count} prompts from bz_agent/prompts directory"
    )


@router.post("/{prompt_name}/activate")
async def activate_prompt(prompt_name: str, version: Optional[int] = None):
    """
    激活指定 Prompt（设置 active=true，其他同名版本的设置为 false）

    Args:
        prompt_name: Prompt 名称
        version: 可选版本号，不指定则激活最新版本

    Returns:
        激活结果消息
    """
    logger.info(f"API: Activate prompt '{prompt_name}' version {version}")

    # 激活 prompt
    success = prompt_store.activate_prompt(prompt_name, version=version)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt '{prompt_name}' not found"
        )

    return {
        "message": f"Prompt '{prompt_name}' activated successfully",
        "prompt_name": prompt_name
    }


@router.get("/reload")
async def reload_prompt_cache():
    """
    重新加载 Prompt 缓存

    当前实现中没有缓存，但此端点预留用于未来扩展
    """
    logger.info("API: Reload prompt cache")

    # 清除可能的缓存并重新加载
    from bz_agent.prompts.template import reload_prompt_cache
    reload_prompt_cache()

    return {
        "message": "Prompt cache reloaded successfully"
    }
