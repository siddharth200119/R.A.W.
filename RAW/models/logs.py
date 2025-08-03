from enum import Enum
from typing import TypedDict, Optional, List
from datetime import datetime
import uuid

class LogLevel(Enum):
    DEBUG = "\033[36m"     # Cyan
    INFO = "\033[32m"      # Green
    WARNING = "\033[33m"   # Yellow
    ERROR = "\033[31m"     # Red

    @property
    def color(self):
        return self.value

    @property
    def name_str(self):
        return self.name

class LogEntry(TypedDict):
    timestamp: datetime
    level: LogLevel
    content: str
    tags: List[str]
    file_path: str
    line_number: int
    traceback: Optional[str]