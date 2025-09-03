from dataclasses import dataclass
from datetime import datetime
import requests

@dataclass
class Ctx:
    now: datetime
    data_api_url: str
    session: requests.Session