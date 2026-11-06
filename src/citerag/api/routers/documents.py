"""Document library: upload, list, delete. Uploads run here because only the backend writes to Chroma."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Response, UploadFile

from citerag.api.deps import Services, get_app_settings, get_services, require_api_key
from citerag.config import Settings
from citerag.errors import FileTooLargeError
from citerag.ingest.loaders import sanitize_name, validate_file
from citerag.ingest.manifest import load_manifest
from citerag.schemas import DocumentInfo, IngestReport

router = APIRouter(prefix="/v1/documents", dependencies=[Depends(require_api_key)])


def _read_valid(files: list[UploadFile], max_mb: int) -> list[tuple[str, bytes]]:
    """Checks every file first, so one bad file in a batch leaves the index unchanged."""
    loaded = []
    for file in files:
        name = sanitize_name(file.filename or "upload")
        if file.size is not None and file.size > max_mb * 1024 * 1024:
            raise FileTooLargeError(f"{name} is larger than {max_mb} MB")
        data = file.file.read()
        validate_file(name, data, max_mb)
        loaded.append((name, data))
    return loaded


@router.post("")
def upload(
    files: Annotated[list[UploadFile], File()],
    services: Annotated[Services, Depends(get_services)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> list[IngestReport]:
    loaded = _read_valid(files, settings.max_upload_mb)
    with services.deps.index_lock:  # captions can take minutes; chat waits, but an upload is rare
        reports = [services.ingestor.ingest_bytes(name, data) for name, data in loaded]
        services.retriever.refresh()
    return reports


@router.get("")
def list_documents(settings: Annotated[Settings, Depends(get_app_settings)]) -> list[DocumentInfo]:
    manifest = load_manifest(settings.index_dir)  # the file only: listing never opens Chroma
    if manifest is None:
        return []
    return [
        DocumentInfo(doc_id=doc_id, doc_name=entry.name, **entry.model_dump(exclude={"name", "sha256"}))
        for doc_id, entry in manifest.docs.items()
    ]


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, services: Annotated[Services, Depends(get_services)]) -> Response:
    with services.deps.index_lock:
        services.ingestor.delete(doc_id)
        services.retriever.refresh()
    return Response(status_code=204)
