from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4



class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    id: UUID # потом при подключении postgre костыль уйдет
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    language: Optional[str] = "ru"



class UserInDB(User):
    hashed_password: str
    