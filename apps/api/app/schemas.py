from pydantic import BaseModel, Field


class ConnectRepositoryRequest(BaseModel):
    repository_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)


class ConnectRepositoryResponse(BaseModel):
    job_id: str
    enqueued: bool
    queue_name: str
    repository_id: str
    status: str
