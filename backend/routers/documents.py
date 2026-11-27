"""
Supporting documents for an application.

Applicants upload to their own applications (for example after an officer
requests documents); the applicant and officers can list and download them.
Files are stored under settings.UPLOAD_DIR/<application id>/ with generated
names; the database keeps the original name and type.
"""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.auth import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.models import Application, AuditLog, Document, User, UserRole
from backend.schemas import DocumentResponse, DocumentsOverview
from backend.serializers import document_requests

router = APIRouter()

DOC_TYPES = {"income_proof", "id_proof", "address_proof", "bank_statement", "employment_proof",
             "guardian_id_proof", "guardian_income_proof", "other"}

# first bytes each allowed type must start with, so a renamed file is refused
SIGNATURES = {
    "application/pdf": (b"%PDF-",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}


def _application_for(db: Session, application_id: int, user: User) -> Application:
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    if user.role == UserRole.APPLICANT and application.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to access this application")
    return application


def _path(doc: Document) -> Path:
    return Path(settings.UPLOAD_DIR) / str(doc.application_id) / doc.stored_name


@router.post("/{application_id}/documents", response_model=DocumentResponse,
             status_code=status.HTTP_201_CREATED)
async def upload_document(
    application_id: int,
    file: UploadFile = File(...),
    doc_type: str = Form("other"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Upload a PDF, PNG or JPEG to your own application"""
    application = _application_for(db, application_id, current_user)
    if application.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Only the applicant can upload documents to an application")
    if doc_type not in DOC_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"doc_type must be one of {sorted(DOC_TYPES)}")

    ext = settings.ALLOWED_UPLOAD_TYPES.get(file.content_type)
    if ext is None:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="Only PDF, PNG and JPEG files are accepted")

    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,  # payload too large; constant name differs across versions
                            detail=f"File is larger than {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    if not data.startswith(SIGNATURES[file.content_type]):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail="File content does not match its declared type")

    doc = Document(
        application_id=application.id,
        uploaded_by=current_user.id,
        doc_type=doc_type,
        original_name=Path(file.filename or "upload").name[:255],
        stored_name=uuid4().hex + ext,
        content_type=file.content_type,
        size_bytes=len(data),
    )
    target = _path(doc)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    db.add(doc)
    db.add(AuditLog(
        application_id=application.id,
        officer_id=None,
        action="document_uploaded",
        justification=f"{doc_type}: {doc.original_name}",
        previous_status=application.status.value,
        new_status=application.status.value,
    ))
    try:
        db.commit()
    except Exception:
        target.unlink(missing_ok=True)
        raise
    db.refresh(doc)
    return doc


@router.get("/{application_id}/documents", response_model=DocumentsOverview)
def list_documents(
    application_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Documents uploaded so far, and what officers have asked for"""
    application = _application_for(db, application_id, current_user)
    return {"requests": document_requests(application), "documents": list(application.documents)}


@router.get("/{application_id}/documents/{document_id}")
def download_document(
    application_id: int,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download one document"""
    _application_for(db, application_id, current_user)
    doc = db.query(Document).filter(
        Document.id == document_id, Document.application_id == application_id
    ).first()
    if not doc or not _path(doc).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return FileResponse(_path(doc), media_type=doc.content_type, filename=doc.original_name)
