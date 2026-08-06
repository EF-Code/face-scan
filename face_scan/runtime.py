"""Runtime diagnostics for OpenCV installations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Dict, List, Optional

import cv2


REQUIRED_OPENCV_APIS = (
    "CascadeClassifier",
    "VideoCapture",
    "VideoWriter",
    "cvtColor",
    "imread",
    "imwrite",
)

OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


@dataclass(frozen=True)
class OpenCVRuntimeReport:
    module_version: str
    module_path: Optional[str]
    distributions: Dict[str, str]
    missing_apis: List[str]

    @property
    def ok(self) -> bool:
        return not self.missing_apis

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def installed_opencv_distributions() -> Dict[str, str]:
    """Return installed OpenCV wheel distributions and their versions."""

    installed: Dict[str, str] = {}
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            installed[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return installed


def inspect_opencv(module: Any = cv2) -> OpenCVRuntimeReport:
    """Inspect the imported cv2 module for APIs required by face-scan."""

    return OpenCVRuntimeReport(
        module_version=str(getattr(module, "__version__", "unknown")),
        module_path=getattr(module, "__file__", None),
        distributions=installed_opencv_distributions(),
        missing_apis=[name for name in REQUIRED_OPENCV_APIS if not hasattr(module, name)],
    )


def require_opencv_api(name: str, module: Any = cv2) -> Any:
    """Return an OpenCV API or raise a diagnostic runtime error."""

    api = getattr(module, name, None)
    if api is not None:
        return api

    report = inspect_opencv(module)
    installed = ", ".join(f"{key}=={value}" for key, value in report.distributions.items()) or "none"
    raise RuntimeError(
        f"The imported OpenCV module does not provide cv2.{name}. "
        f"Loaded cv2 {report.module_version} from {report.module_path}; "
        f"installed OpenCV distributions: {installed}. "
        "Install exactly one compatible OpenCV wheel and rerun doctor.py."
    )
