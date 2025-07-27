from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel


class Image(BaseModel):
    data: str

    @classmethod
    def from_path(cls, path: Path) -> "Image":
        with open(path, "rb") as f:
            return cls.from_binary(f.read())

    @classmethod
    def from_np_array(cls, array: np.ndarray) -> "Image":
        _, buffer = cv2.imencode(".png", array)
        return cls.from_binary(buffer.tobytes())

    @classmethod
    def from_base64(cls, data: str) -> "Image":
        return cls(data=data)

    @classmethod
    def from_binary(cls, data: bytes) -> "Image":
        return cls(data=base64.b64encode(data).decode("utf-8"))

    def to_path(self, path: Path) -> None:
        with open(path, "wb") as f:
            f.write(self.to_binary())

    def to_base64(self) -> str:
        return self.data

    def to_binary(self) -> bytes:
        return base64.b64decode(self.data)

    def to_np_array(self) -> np.ndarray:
        img_binary = self.to_binary()
        np_arr = np.frombuffer(img_binary, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
