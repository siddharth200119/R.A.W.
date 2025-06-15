import asyncio
import aiofiles
from typing import Optional, Any
from pathlib import Path
from enum import Enum
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
import logging

class LogLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"
    CRITICAL = "critical"

class Logger:
    def __init__(self, name: str = "async_logger", log_file: Optional[Path] = None, log_time: bool = True, use_colors: bool = True) -> None:
        self.name = name
        self.save_to_file: bool = log_file is not None
        if self.save_to_file:
            self.log_file: Path = log_file
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_time: bool = log_time
        self.use_colors: bool = use_colors
        self._lock = asyncio.Lock()
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        console = Console(color_system="auto" if use_colors else None)
        rich_handler = RichHandler(
            console=console, 
            show_path=False, 
            markup=True,
            show_time=log_time,
            omit_repeated_times=False,
            show_level=True,
            enable_link_path=False
        )
        
        formatter = logging.Formatter("%(message)s")
        rich_handler.setFormatter(formatter)
        self.logger.addHandler(rich_handler)

    def _format_file_message(self, content: Any, level: str) -> str:
        """Format the log message for file output (without colors)."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.log_time else ""
        level_str = f"[{level.upper()}]"
        content = repr(content) if hasattr(content, '__repr__') else content
        if self.log_time:
            return f"{timestamp} {level_str} {content}\n"
        else:
            return f"{level_str} {content}\n"

    async def _log(self, level: str, content: Any) -> None:
        """Internal async logging method that handles both console and file output."""
        async with self._lock:
            # Log to console using Rich (via standard logging)
            log_method = getattr(self.logger, level.lower())
            log_method(str(content))
            
            # Write to file if configured
            if self.save_to_file:
                try:
                    file_message = self._format_file_message(content, level)
                    async with aiofiles.open(self.log_file, mode='a', encoding='utf-8') as f:
                        await f.write(file_message)
                except Exception as e:
                    self.logger.error(f"Failed to write to log file: {e}")

    def _log_sync(self, level: str, content: Any) -> None:
        """Internal sync logging method that handles both console and file output."""
        # Log to console using Rich (via standard logging)
        log_method = getattr(self.logger, level.lower())
        log_method(str(content))
        
        # Write to file if configured
        if self.save_to_file:
            try:
                file_message = self._format_file_message(content, level)
                with open(self.log_file, mode='a', encoding='utf-8') as f:
                    f.write(file_message)
            except Exception as e:
                self.logger.error(f"Failed to write to log file: {e}")

    async def info(self, content: Any) -> None:
        """Log an info message asynchronously."""
        await self._log("info", content)

    def info_sync(self, content: Any) -> None:
        """Log an info message synchronously."""
        self._log_sync("info", content)

    async def warning(self, content: Any) -> None:
        """Log a warning message asynchronously."""
        await self._log("warning", content)

    def warning_sync(self, content: Any) -> None:
        """Log a warning message synchronously."""
        self._log_sync("warning", content)

    async def error(self, content: Any) -> None:
        """Log an error message asynchronously."""
        await self._log("error", content)

    def error_sync(self, content: Any) -> None:
        """Log an error message synchronously."""
        self._log_sync("error", content)

    async def debug(self, content: Any) -> None:
        """Log a debug message asynchronously."""
        await self._log("debug", content)

    def debug_sync(self, content: Any) -> None:
        """Log a debug message synchronously."""
        self._log_sync("debug", content)

    async def critical(self, content: Any) -> None:
        """Log a critical message asynchronously."""
        await self._log("critical", content)

    def critical_sync(self, content: Any) -> None:
        """Log a critical message synchronously."""
        self._log_sync("critical", content)

    async def log(self, content: Any, level: LogLevel) -> None:
        """Generic async log method that accepts a LogLevel enum."""
        await self._log(level.value, content)

    def log_sync(self, content: Any, level: LogLevel) -> None:
        """Generic sync log method that accepts a LogLevel enum."""
        self._log_sync(level.value, content)

    async def log_str(self, level: str, content: Any) -> None:
        """Generic async log method that accepts a level string."""
        await self._log(level, content)

    def log_str_sync(self, level: str, content: Any) -> None:
        """Generic sync log method that accepts a level string."""
        self._log_sync(level, content)

    async def close(self) -> None:
        """Close the logger and cleanup handlers asynchronously."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def close_sync(self) -> None:
        """Close the logger and cleanup handlers synchronously."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

# # Example usage
# async def main():
#     # Create logger with Rich formatting, file output, and timestamps
#     logger = Logger("my_app", Path("app.log"), log_time=True, use_colors=True)
    
#     # Async logging
#     await logger.info("🚀 Application started successfully")
#     await logger.warning("⚠️ This is a warning message")
#     await logger.error("❌ An error occurred during processing")
#     await logger.debug("🐛 Debug information for troubleshooting")
#     await logger.critical("🔥 Critical system error - immediate attention required!")
    
#     # Sync logging
#     logger.info_sync("🚀 Sync: Application started successfully")
#     logger.warning_sync("⚠️ Sync: This is a warning message")
#     logger.error_sync("❌ Sync: An error occurred during processing")
#     logger.debug_sync("🐛 Sync: Debug information for troubleshooting")
#     logger.critical_sync("🔥 Sync: Critical system error - immediate attention required!")
    
#     # Create console-only logger without timestamps
#     console_logger = Logger("console_app", log_time=False, use_colors=True)
#     await console_logger.info("✨ Console only message with rich formatting")
#     console_logger.info_sync("✨ Sync: Console only message with rich formatting")
    
#     # Create logger without colors (useful for CI/CD environments)
#     plain_logger = Logger("plain_app", use_colors=False)
#     await plain_logger.info("Plain message without colors")
#     plain_logger.info_sync("Sync: Plain message without colors")
    
#     # Demonstrate logging various data types with markup
#     await logger.info("[bold blue]User login:[/bold blue] john_doe")
#     logger.info_sync("[bold blue]Sync: User login:[/bold blue] john_doe")
#     await logger.error({"error_code": 500, "message": "Internal server error"})
#     logger.error_sync({"error_code": 500, "message": "Sync: Internal server error"})
#     await logger.debug(["step1", "step2", "step3"])
#     logger.debug_sync(["step1", "step2", "step3"])
#     await logger.warning("[yellow]Memory usage:[/yellow] 85%")
#     logger.warning_sync("[yellow]Sync: Memory usage:[/yellow] 85%")
    
#     await logger.close()
#     console_logger.close_sync()
#     plain_logger.close_sync()

# if __name__ == "__main__":
#     asyncio.run(main())