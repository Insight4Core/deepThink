from typing import Optional, List
from pydantic import BaseModel, Field


class ClarifierOutput(BaseModel):
    """意图澄清节点的结构化输出"""
    is_clear: bool = Field(
        description="用户问题是否已经足够清晰，可以直接进入生成阶段"
    )
    response: str = Field(
        description="如果 is_clear 为 true，这里是重构后的清晰问题描述；如果为 false，这里是向用户追问的内容"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="简要说明你判断问题是否清晰的理由"
    )


class ReviewFeedback(BaseModel):
    """单个 Reviewer 的结构化反馈"""
    score: int = Field(
        ge=0,
        le=10,
        description="从 0 到 10 对当前草稿在该审查维度下的评分，10 表示无可挑剔"
    )
    comments: str = Field(
        description="具体、可执行的修改建议；如果 score 为 10，可写 'PASS'"
    )
    passed: bool = Field(
        description="该维度是否通过审查。通常 score >= 8 视为通过"
    )


class JudgeOutput(BaseModel):
    """LLM-as-Judge 评分输出"""
    score: int = Field(
        ge=0,
        le=10,
        description="0-10 的综合评分"
    )
    reasoning: str = Field(
        description="打分的简要理由"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="回答的优点"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="回答的不足或改进建议"
    )
