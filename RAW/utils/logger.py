import sys
import threading
import queue
import uuid
import json
import traceback
import inspect
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from RAW.models import LogEntry, LogLevel

class Logger:
    def __init__(self, log_file: Optional[Path] = None, queue_size: int = 1000):
        self.queue: queue.Queue[LogEntry] = queue.Queue(maxsize=queue_size)
        self.log_file = log_file
        self.running: bool = True
        self.thread = threading.Thread(target=self.process, daemon=True)
        self.thread.start()

    def _get_caller_info(self, stack_offset: int = 3):
        """Get file path and line number of the caller"""
        try:
            frame = inspect.currentframe()
            for _ in range(stack_offset):
                frame = frame.f_back
                if frame is None:
                    return "<unknown>", 0
            
            file_path = frame.f_code.co_filename
            line_number = frame.f_lineno
            return file_path, line_number
        except:
            return "<unknown>", 0

    def put(self, level: LogLevel, content: str, tags: Optional[List[str]] = None, 
            track_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None,
            error: Optional[Exception] = None):
        
        file_path, line_number = self._get_caller_info(stack_offset=3)
        
        # Handle traceback for errors
        tb = None
        if error is not None:
            tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        elif level == LogLevel.ERROR and not error:
            # If it's an error level but no exception passed, try to get current traceback
            tb = ''.join(traceback.format_stack())
        
        entry: LogEntry = {
            "timestamp": timestamp or datetime.now(),
            "track_id": track_id or uuid.uuid4(),
            "level": level,
            "content": content,
            "tags": tags or [],
            "file_path": file_path,
            "line_number": line_number,
            "traceback": tb
        }
        
        try:
            self.queue.put(entry, timeout=1)
        except queue.Full:
            sys.stderr.write("⚠️ Logger queue is full. Dropping log entry.\n")
            sys.stderr.flush()

    def stop(self):
        self.running = False
        self.queue.join()
        self.thread.join()

    def process(self):
        while self.running or not self.queue.empty():
            try:
                entry = self.queue.get(timeout=1)
            except queue.Empty:
                continue

            color = entry['level'].color
            reset = "\033[0m"
            timestamp = entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            track_id = entry['track_id']
            level = entry['level'].name_str
            content = entry['content']
            tags = ', '.join(entry['tags'])
            file_path = Path(entry['file_path']).name  # Just filename for readability
            line_number = entry['line_number']

            # Print to stdout
            sys.stdout.write(
                f"{color}[{timestamp}] [{level}] {content} "
                f"({file_path}:{line_number}, track_id={track_id}, tags=[{tags}]){reset}\n"
            )
            
            # Print traceback if present
            if entry['traceback']:
                sys.stdout.write(f"{color}Traceback:\n{entry['traceback']}{reset}")
            
            sys.stdout.flush()

            # Log to file
            if self.log_file:
                with self.log_file.open('a') as f:
                    f.write(json.dumps(entry, default=str) + "\n")
                    f.flush()

            self.queue.task_done()

    def debug(self, content: str, tags: Optional[List[str]] = None, 
              track_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None):
        """Log a debug message"""
        file_path, line_number = self._get_caller_info(stack_offset=2)
        entry: LogEntry = {
            "timestamp": timestamp or datetime.now(),
            "track_id": track_id or uuid.uuid4(),
            "level": LogLevel.DEBUG,
            "content": content,
            "tags": tags or [],
            "file_path": file_path,
            "line_number": line_number,
            "traceback": None
        }
        try:
            self.queue.put(entry, timeout=1)
        except queue.Full:
            sys.stderr.write("⚠️ Logger queue is full. Dropping log entry.\n")
            sys.stderr.flush()

    def info(self, content: str, tags: Optional[List[str]] = None, 
             track_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None):
        """Log an info message"""
        file_path, line_number = self._get_caller_info(stack_offset=2)
        entry: LogEntry = {
            "timestamp": timestamp or datetime.now(),
            "track_id": track_id or uuid.uuid4(),
            "level": LogLevel.INFO,
            "content": content,
            "tags": tags or [],
            "file_path": file_path,
            "line_number": line_number,
            "traceback": None
        }
        try:
            self.queue.put(entry, timeout=1)
        except queue.Full:
            sys.stderr.write("⚠️ Logger queue is full. Dropping log entry.\n")
            sys.stderr.flush()

    def warning(self, content: str, tags: Optional[List[str]] = None, 
                track_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None):
        """Log a warning message"""
        file_path, line_number = self._get_caller_info(stack_offset=2)
        entry: LogEntry = {
            "timestamp": timestamp or datetime.now(),
            "track_id": track_id or uuid.uuid4(),
            "level": LogLevel.WARNING,
            "content": content,
            "tags": tags or [],
            "file_path": file_path,
            "line_number": line_number,
            "traceback": None
        }
        try:
            self.queue.put(entry, timeout=1)
        except queue.Full:
            sys.stderr.write("⚠️ Logger queue is full. Dropping log entry.\n")
            sys.stderr.flush()

    def error(self, content: str, tags: Optional[List[str]] = None, 
              track_id: Optional[uuid.UUID] = None, timestamp: Optional[datetime] = None,
              error: Optional[Exception] = None):
        """Log an error message with optional exception traceback"""
        file_path, line_number = self._get_caller_info(stack_offset=2)
        
        # Handle traceback for errors
        tb = None
        if error is not None:
            tb = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
        entry: LogEntry = {
            "timestamp": timestamp or datetime.now(),
            "track_id": track_id or uuid.uuid4(),
            "level": LogLevel.ERROR,
            "content": content,
            "tags": tags or [],
            "file_path": file_path,
            "line_number": line_number,
            "traceback": tb
        }
        try:
            self.queue.put(entry, timeout=1)
        except queue.Full:
            sys.stderr.write("⚠️ Logger queue is full. Dropping log entry.\n")
            sys.stderr.flush()