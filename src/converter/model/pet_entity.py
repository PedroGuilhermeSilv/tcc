import uuid
from uuid import UUID
from pydantic import BaseModel, Field

class Pet(BaseModel):
    id: UUID = Field(default_factory=uuid.uuid4)
    name: str
    pet_type: str
    affected_limb: str 