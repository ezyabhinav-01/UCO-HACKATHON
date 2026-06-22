"""
app/api/v1/endpoints/users.py

User management endpoints:

    POST   /api/v1/users           - create a new enrollable user
    GET    /api/v1/users/{id}       - fetch a user and enrollment status
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.session import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserDetailRead, UserRead
from app.utils.exceptions import UserAlreadyExistsError, UserNotFoundError

log = get_logger(__name__)

router = APIRouter()


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description=(
        "Creates a new user record. A user must exist before they can be "
        "enrolled (POST /api/v1/enroll) or verified (POST /api/v1/verify)."
    ),
)
async def create_user(
    payload: UserCreate, db: AsyncSession = Depends(get_db)
) -> UserRead:
    repo = UserRepository(db)

    existing = await repo.get_by_email(payload.email)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=UserAlreadyExistsError(payload.email).message,
        )

    try:
        user = await repo.create(name=payload.name, email=payload.email)
        await db.commit()
        return UserRead.model_validate(user)
    except Exception as exc:  # noqa: BLE001
        await db.rollback()
        log.exception("Failed to create user")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user.",
        ) from exc


@router.get(
    "/users/{user_id}",
    response_model=UserDetailRead,
    status_code=status.HTTP_200_OK,
    summary="Get a user and their enrollment status",
)
async def get_user(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> UserDetailRead:
    repo = UserRepository(db)
    user: User | None = await repo.get_by_id_with_voiceprint(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=UserNotFoundError(str(user_id)).message,
        )

    voiceprint = user.voiceprint

    return UserDetailRead(
        id=user.id,
        name=user.name,
        email=user.email,
        created_at=user.created_at,
        is_enrolled=voiceprint is not None,
        recording_count=voiceprint.recording_count if voiceprint else 0,
        embedding_dimension=len(voiceprint.embedding) if voiceprint else None,
        voiceprint_updated_at=voiceprint.updated_at if voiceprint else None,
    )
