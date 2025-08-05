from typing import Literal, List, Optional
import re
import json

from RAW.models import Message
from RAW.utils import Logger
from RAW.agentic.llms import LLM

class ShortTermMemory():
    def __init__(
            self, 
            llm: Optional[LLM] = None,
            logger: Logger = Logger(),
            actionable_unit: Literal['length', 'context'] = 'length',
            actionable_quantity: int = 10,
            summarize_to_system_prompt: bool = False
        ):
        self.llm = llm
        self.actionable_unit = actionable_unit
        self.actionable_quantity = actionable_quantity
        self.logger = logger
        self.summarize_to_system_prompt = summarize_to_system_prompt

    def _extract_existing_summary(self, system_content: str) -> tuple[str, Optional[str]]:
        """
        Extract existing summary from system content and return clean content + summary.
        Returns: (content_without_summary, existing_summary)
        """
        # Pattern to match summary sections
        summary_pattern = r'\n*Summary of the earliest \d+ (?:messages? )?(?:which were removed to save context|that were compressed):\n*(.*?)(?=\n\n|\Z)'
        
        match = re.search(summary_pattern, system_content, re.DOTALL | re.IGNORECASE)
        if match:
            existing_summary = match.group(1).strip()
            # Remove the summary section from content
            clean_content = re.sub(summary_pattern, '', system_content, flags=re.DOTALL | re.IGNORECASE).strip()
            return clean_content, existing_summary
        
        return system_content.strip(), None

    async def _generate_summary(self, messages: List[Message], existing_summary: Optional[str] = None) -> str:
        """Generate or merge summaries for removed messages."""
        messages_json = json.dumps([message.model_dump() for message in messages], indent=2)
        
        if existing_summary:
            prompt = f"""Merge and create a comprehensive summary that preserves ALL context and details:

EXISTING SUMMARY:
{existing_summary}

NEW MESSAGES TO ADD:
{messages_json}

Requirements:
- Include ALL topics, subtopics, and details discussed
- Preserve specific examples, code snippets, configurations, and technical details
- Maintain chronological flow and conversation context
- Include user preferences, decisions made, and problem-solving approaches
- Keep all names, numbers, file paths, URLs, and specific references
- Preserve the nuance of discussions and any important clarifications
- Include background context and reasoning behind decisions
- Maintain emotional context and user's state of mind if relevant

Provide only the comprehensive merged summary without any preamble or explanation."""
        else:
            prompt = f"""Create a comprehensive summary of the following LLM conversation that preserves ALL context and details:

{messages_json}

Requirements:
- Include ALL topics, subtopics, and details discussed
- Preserve specific examples, code snippets, configurations, and technical details
- Maintain chronological flow and conversation context
- Include user preferences, decisions made, and problem-solving approaches
- Keep all names, numbers, file paths, URLs, and specific references
- Preserve the nuance of discussions and any important clarifications
- Include background context and reasoning behind decisions
- Maintain emotional context and user's state of mind if relevant
- Do not omit any information, even if it seems minor or tangential

Provide only the comprehensive summary without any preamble or explanation."""
        
        response = await self.llm.generate(prompt=prompt)
        return response.content.strip()

    async def __call__(self, messages: List[Message]) -> tuple[List[Message], List[Message]]:
        if not messages:
            return messages, []
    
        system_message: Optional[Message] = None
        preserved_messages: List[Message] = messages.copy()
        removed_messages: List[Message] = []
        
        if messages[0].role == 'system':
            system_message = preserved_messages.pop(0)

        if self.actionable_unit == 'length':
            if len(preserved_messages) >= self.actionable_quantity:
                removed_messages = preserved_messages[:-self.actionable_quantity]
                preserved_messages = preserved_messages[-self.actionable_quantity:]
                while preserved_messages and preserved_messages[0].role != 'user':
                    removed_messages.append(preserved_messages.pop(0))

                if len(preserved_messages) == 0:
                    self.logger.warning(f'No user message found while compressing messages using Unit = {self.actionable_unit}, quantity = {self.actionable_quantity}')
                    
        else:
            current_tokens = self._count_tokens(messages=preserved_messages)
            
            if current_tokens >= self.actionable_quantity:
                while (preserved_messages and 
                       self._count_tokens(messages=preserved_messages) >= self.actionable_quantity):
                    removed_messages.append(preserved_messages.pop(0))
                
                while preserved_messages and preserved_messages[0].role != 'user':
                    removed_messages.append(preserved_messages.pop(0))
                
                if len(preserved_messages) == 0:
                    self.logger.warning(f'No user message found while compressing messages using Unit = {self.actionable_unit}, quantity = {self.actionable_quantity}')

        result: List[Message] = []
        if system_message:
            if self.summarize_to_system_prompt and self.llm and len(removed_messages) > 0:
                # Extract existing summary if present
                clean_content, existing_summary = self._extract_existing_summary(system_message.content)
                
                # Generate new summary (merging with existing if present)
                new_summary = await self._generate_summary(removed_messages, existing_summary)
                
                # Update system message content
                total_removed = len(removed_messages) + (len(existing_summary.split()) if existing_summary else 0)
                system_message.content = f"""{clean_content}

Summary of the earliest {total_removed} messages which were removed to save context:

{new_summary}

"""
            result.append(system_message)
        result.extend(preserved_messages)
        
        return result

    def _count_tokens(self, messages: List[Message]) -> int:
        text = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )
        return int(len(re.findall(r"\S+", text)) * 1.3)