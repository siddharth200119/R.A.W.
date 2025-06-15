import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from RAW import Logger, LogLevel


class TestLogger:
    """Test suite for the Logger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def log_file(self, temp_dir):
        """Create a temporary log file path."""
        return temp_dir / "test.log"

    @pytest.fixture
    def logger(self, log_file):
        """Create a logger instance for testing."""
        return Logger("test_logger", log_file, log_time=True, use_colors=False)

    @pytest.fixture
    def console_logger(self):
        """Create a console-only logger for testing."""
        return Logger("console_test", log_time=False, use_colors=False)

    @pytest.mark.asyncio
    async def test_logger_initialization(self, log_file):
        """Test logger initialization with different configurations."""
        # Test with file output
        logger = Logger("test", log_file, log_time=True, use_colors=True)
        assert logger.name == "test"
        assert logger.save_to_file is True
        assert logger.log_file == log_file
        assert logger.log_time is True
        assert logger.use_colors is True
        await logger.close()

        # Test console-only logger
        console_logger = Logger("console_test", log_time=False, use_colors=False)
        assert console_logger.name == "console_test"
        assert console_logger.save_to_file is False
        assert console_logger.log_time is False
        assert console_logger.use_colors is False
        await console_logger.close()

    @pytest.mark.asyncio
    async def test_directory_creation(self, temp_dir):
        """Test that parent directories are created for log files."""
        nested_log_file = temp_dir / "nested" / "deep" / "test.log"
        logger = Logger("test", nested_log_file)
        
        assert nested_log_file.parent.exists()
        await logger.close()

    @pytest.mark.asyncio
    async def test_info_logging(self, logger, log_file):
        """Test async info level logging."""
        test_message = "This is an info message"
        await logger.info(test_message)
        
        # Check file content
        content = log_file.read_text()
        assert "[INFO]" in content
        assert test_message in content
        await logger.close()

    def test_info_sync_logging(self, logger, log_file):
        """Test sync info level logging."""
        test_message = "This is a sync info message"
        logger.info_sync(test_message)
        
        # Check file content
        content = log_file.read_text()
        assert "[INFO]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_warning_logging(self, logger, log_file):
        """Test async warning level logging."""
        test_message = "This is a warning message"
        await logger.warning(test_message)
        
        content = log_file.read_text()
        assert "[WARNING]" in content
        assert test_message in content
        await logger.close()

    def test_warning_sync_logging(self, logger, log_file):
        """Test sync warning level logging."""
        test_message = "This is a sync warning message"
        logger.warning_sync(test_message)
        
        content = log_file.read_text()
        assert "[WARNING]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_error_logging(self, logger, log_file):
        """Test async error level logging."""
        test_message = "This is an error message"
        await logger.error(test_message)
        
        content = log_file.read_text()
        assert "[ERROR]" in content
        assert test_message in content
        await logger.close()

    def test_error_sync_logging(self, logger, log_file):
        """Test sync error level logging."""
        test_message = "This is a sync error message"
        logger.error_sync(test_message)
        
        content = log_file.read_text()
        assert "[ERROR]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_debug_logging(self, logger, log_file):
        """Test async debug level logging."""
        test_message = "This is a debug message"
        await logger.debug(test_message)
        
        content = log_file.read_text()
        assert "[DEBUG]" in content
        assert test_message in content
        await logger.close()

    def test_debug_sync_logging(self, logger, log_file):
        """Test sync debug level logging."""
        test_message = "This is a sync debug message"
        logger.debug_sync(test_message)
        
        content = log_file.read_text()
        assert "[DEBUG]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_critical_logging(self, logger, log_file):
        """Test async critical level logging."""
        test_message = "This is a critical message"
        await logger.critical(test_message)
        
        content = log_file.read_text()
        assert "[CRITICAL]" in content
        assert test_message in content
        await logger.close()

    def test_critical_sync_logging(self, logger, log_file):
        """Test sync critical level logging."""
        test_message = "This is a sync critical message"
        logger.critical_sync(test_message)
        
        content = log_file.read_text()
        assert "[CRITICAL]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_log_with_enum(self, logger, log_file):
        """Test async logging using LogLevel enum."""
        test_message = "Enum log message"
        await logger.log(test_message, LogLevel.INFO)
        
        content = log_file.read_text()
        assert "[INFO]" in content
        assert test_message in content
        await logger.close()

    def test_log_sync_with_enum(self, logger, log_file):
        """Test sync logging using LogLevel enum."""
        test_message = "Sync enum log message"
        logger.log_sync(test_message, LogLevel.INFO)
        
        content = log_file.read_text()
        assert "[INFO]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_log_with_string(self, logger, log_file):
        """Test async logging using string level."""
        test_message = "String log message"
        await logger.log_str("warning", test_message)
        
        content = log_file.read_text()
        assert "[WARNING]" in content
        assert test_message in content
        await logger.close()

    def test_log_sync_with_string(self, logger, log_file):
        """Test sync logging using string level."""
        test_message = "Sync string log message"
        logger.log_str_sync("warning", test_message)
        
        content = log_file.read_text()
        assert "[WARNING]" in content
        assert test_message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_timestamp_in_file(self, logger, log_file):
        """Test that timestamps are included in file output."""
        await logger.info("Timestamped message")
        
        content = log_file.read_text()
        # Check for timestamp format YYYY-MM-DD HH:MM:SS
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, content)
        await logger.close()

    def test_timestamp_in_file_sync(self, logger, log_file):
        """Test that timestamps are included in file output for sync logging."""
        logger.info_sync("Sync timestamped message")
        
        content = log_file.read_text()
        # Check for timestamp format YYYY-MM-DD HH:MM:SS
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert re.search(timestamp_pattern, content)
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_no_timestamp_option(self, temp_dir):
        """Test logger without timestamps."""
        log_file = temp_dir / "no_timestamp.log"
        logger = Logger("test", log_file, log_time=False)
        
        await logger.info("No timestamp message")
        
        content = log_file.read_text()
        assert "[INFO]" in content
        assert "No timestamp message" in content
        # Should not contain timestamp
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert not re.search(timestamp_pattern, content)
        
        await logger.close()

    def test_no_timestamp_option_sync(self, temp_dir):
        """Test sync logger without timestamps."""
        log_file = temp_dir / "no_timestamp.log"
        logger = Logger("test", log_file, log_time=False)
        
        logger.info_sync("No timestamp sync message")
        
        content = log_file.read_text()
        assert "[INFO]" in content
        assert "No timestamp sync message" in content
        # Should not contain timestamp
        import re
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        assert not re.search(timestamp_pattern, content)
        
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_console_only_logger(self, console_logger):
        """Test async console-only logger (no file output)."""
        await console_logger.info("Console only message")
        await console_logger.error("Console error")
        await console_logger.debug("Console debug")
        await console_logger.close()

    def test_console_only_logger_sync(self, console_logger):
        """Test sync console-only logger (no file output)."""
        console_logger.info_sync("Sync console only message")
        console_logger.error_sync("Sync console error")
        console_logger.debug_sync("Sync console debug")
        console_logger.close_sync()

    @pytest.mark.asyncio
    async def test_multiple_log_messages(self, logger, log_file):
        """Test multiple async log messages are written correctly."""
        messages = [
            ("info", "First message"),
            ("warning", "Second message"),
            ("error", "Third message")
        ]
        
        for level, message in messages:
            await getattr(logger, level)(message)
        
        content = log_file.read_text()
        for level, message in messages:
            assert f"[{level.upper()}]" in content
            assert message in content
        await logger.close()

    def test_multiple_log_messages_sync(self, logger, log_file):
        """Test multiple sync log messages are written correctly."""
        messages = [
            ("info_sync", "Sync first message"),
            ("warning_sync", "Sync second message"),
            ("error_sync", "Sync third message")
        ]
        
        for level, message in messages:
            getattr(logger, level)(message)
        
        content = log_file.read_text()
        for level, message in messages:
            assert f"[{level.replace('_sync', '').upper()}]" in content
            assert message in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_concurrent_logging(self, logger, log_file):
        """Test concurrent async logging operations."""
        async def log_messages(prefix, count):
            for i in range(count):
                await logger.info(f"{prefix} message {i}")
        
        # Run multiple concurrent logging tasks
        await asyncio.gather(
            log_messages("Task1", 5),
            log_messages("Task2", 5),
            log_messages("Task3", 5)
        )
        
        content = log_file.read_text()
        lines = content.strip().split('\n')
        # Should have 15 log entries
        assert len(lines) == 15
        
        # Check that all messages are present
        for i in range(5):
            assert f"Task1 message {i}" in content
            assert f"Task2 message {i}" in content
            assert f"Task3 message {i}" in content
        await logger.close()

    def test_complex_data_types_sync(self, logger, log_file):
        """Test sync logging complex data types."""
        test_data = {
            "dict": {"key": "value", "number": 42},
            "list": [1, 2, 3, "string"],
            "tuple": (1, "tuple", True),
            "number": 3.14159
        }
        
        for data_type, data in test_data.items():
            logger.info_sync(data)
        
        content = log_file.read_text()
        assert "{'key': 'value', 'number': 42}" in content
        assert "[1, 2, 3, 'string']" in content
        assert "3.14159" in content
        logger.close_sync()

    @pytest.mark.asyncio
    async def test_complex_data_types(self, logger, log_file):
        """Test async logging complex data types."""
        test_data = {
            "dict": {"key": "value", "number": 42},
            "list": [1, 2, 3, "string"],
            "tuple": (1, "tuple", True),
            "number": 3.14159
        }
        
        for data_type, data in test_data.items():
            await logger.info(data)
        
        content = log_file.read_text()
        assert "{'key': 'value', 'number': 42}" in content
        assert "[1, 2, 3, 'string']" in content
        assert "3.14159" in content
        await logger.close()

    @pytest.mark.asyncio
    async def test_file_write_error_handling(self, temp_dir):
        """Test error handling when file write fails."""
        # Create a logger with a file in a non-existent directory that can't be created
        if Path("/root").exists():  # Skip on systems where /root might be writable
            pytest.skip("Skipping on systems where /root might be writable")
        
        invalid_log_file = Path("/root/nonexistent/test.log")
        
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                Logger("test", invalid_log_file)

    @pytest.mark.asyncio
    async def test_logger_cleanup(self, log_file):
        """Test async logger cleanup and handler removal."""
        logger = Logger("test_cleanup", log_file)
        
        # Verify handlers are added
        assert len(logger.logger.handlers) > 0
        
        # Close the logger
        await logger.close()
        
        # Verify handlers are removed
        assert len(logger.logger.handlers) == 0

    def test_logger_cleanup_sync(self, log_file):
        """Test sync logger cleanup and handler removal."""
        logger = Logger("test_cleanup", log_file)
        
        # Verify handlers are added
        assert len(logger.logger.handlers) > 0
        
        # Close the logger
        logger.close_sync()
        
        # Verify handlers are removed
        assert len(logger.logger.handlers) == 0

    def test_log_level_enum(self):
        """Test LogLevel enum values."""
        assert LogLevel.INFO.value == "info"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.CRITICAL.value == "critical"

    @pytest.mark.asyncio
    async def test_rich_markup_in_console(self, console_logger):
        """Test that Rich markup works in async console output."""
        await console_logger.info("[bold]Bold text[/bold]")
        await console_logger.warning("[yellow]Yellow warning[/yellow]")
        await console_logger.error("[red]Red error[/red]")
        await console_logger.close()

    def test_rich_markup_in_console_sync(self, console_logger):
        """Test that Rich markup works in sync console output."""
        console_logger.info_sync("[bold]Sync bold text[/bold]")
        console_logger.warning_sync("[yellow]Sync yellow warning[/yellow]")
        console_logger.error_sync("[red]Sync red error[/red]")
        console_logger.close_sync()

    @pytest.mark.asyncio
    async def test_logger_with_same_name(self, log_file):
        """Test creating multiple loggers with the same name."""
        logger1 = Logger("same_name", log_file)
        logger2 = Logger("same_name", log_file)
        
        # Both should work without conflicts
        await logger1.info("Logger 1 message")
        await logger2.info("Logger 2 message")
        
        await logger1.close()
        await logger2.close()

    def test_logger_with_same_name_sync(self, log_file):
        """Test sync logging with multiple loggers with the same name."""
        logger1 = Logger("same_name", log_file)
        logger2 = Logger("same_name", log_file)
        
        # Both should work without conflicts
        logger1.info_sync("Sync logger 1 message")
        logger2.info_sync("Sync logger 2 message")
        
        logger1.close_sync()
        logger2.close_sync()


# Pytest configuration and fixtures for the entire test suite
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Example of running specific tests
if __name__ == "__main__":
    # Run with: python -m pytest test_logger.py -v
    pytest.main([__file__, "-v"])