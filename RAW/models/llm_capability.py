from enum import Enum

class LLMCapability(Enum):
    THINKING="thinking"
    TOOLS="tools"
    VISION="vision"
    COMPLETION="completion"