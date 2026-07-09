from __future__ import annotations
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_MODE = os.getenv("STORAGE_MODE", "local").strip().lower()
LOCAL_STORAGE_ROOT = Path(os.getenv("LOCAL_STORAGE_ROOT", PROJECT_ROOT / "data"))
OCI_BUCKET_NAME = os.getenv("OCI_BUCKET_NAME", "").strip()
OCI_NAMESPACE = os.getenv("OCI_NAMESPACE", "").strip()
OCI_CONFIG_FILE = os.getenv("OCI_CONFIG_FILE", "").strip() or None
OCI_CONFIG_PROFILE = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT").strip() or "DEFAULT"
OCI_OBJECT_PREFIX = os.getenv("OCI_OBJECT_PREFIX", "ai-cloud-pipeline").strip().strip("/")


def _relative_to_project(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def save_bytes(relative_path: str, content: bytes) -> Path:
    path = LOCAL_STORAGE_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def save_text(relative_path: str, content: str, encoding: str = "utf-8") -> Path:
    path = LOCAL_STORAGE_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)
    return path


def object_name_for_path(path: str | Path) -> str:
    relative_path = _relative_to_project(path)
    return f"{OCI_OBJECT_PREFIX}/{relative_path}" if OCI_OBJECT_PREFIX else relative_path


def upload_to_object_storage(file_path: str | Path, object_name: str | None = None) -> str:
    if not OCI_BUCKET_NAME:
        raise RuntimeError("OCI_BUCKET_NAME 환경변수가 설정되어 있지 않습니다.")

    try:
        import oci
    except ImportError as error:
        raise RuntimeError("OCI Object Storage 업로드에는 oci 패키지가 필요합니다.") from error

    if OCI_CONFIG_FILE:
        config = oci.config.from_file(OCI_CONFIG_FILE, OCI_CONFIG_PROFILE)
    else:
        config = oci.config.from_file(profile_name=OCI_CONFIG_PROFILE)
    object_storage = oci.object_storage.ObjectStorageClient(config)
    namespace = OCI_NAMESPACE or object_storage.get_namespace().data
    object_name = object_name or object_name_for_path(file_path)

    with open(file_path, "rb") as file_obj:
        object_storage.put_object(namespace, OCI_BUCKET_NAME, object_name, file_obj)

    return f"oci://{OCI_BUCKET_NAME}/{object_name}"


def maybe_upload_to_object_storage(file_path: str | Path, object_name: str | None = None) -> str:
    if STORAGE_MODE not in {"oci", "object_storage", "cloud"}:
        return ""
    return upload_to_object_storage(file_path, object_name)


def build_storage_metadata(file_path: str | Path, storage_uri: str = "") -> dict:
    return {
        "storage_mode": STORAGE_MODE,
        "local_path": _relative_to_project(file_path),
        "object_storage_uri": storage_uri,
        "object_storage_bucket": OCI_BUCKET_NAME,
    }
