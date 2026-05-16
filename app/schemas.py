from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    session_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class ClearResponse(BaseModel):
    session_id: str
    cleared: bool
