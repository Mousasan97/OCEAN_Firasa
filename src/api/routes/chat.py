"""
Chat routes for AI Personality Coach

Provides API endpoints for conversational AI coaching based on personality analysis.
Supports both regular and streaming (SSE) responses.
Includes CV upload for enhanced job matching.
"""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator

from src.services.ai_agent_service import get_personality_coach, get_last_job_search_results, peek_job_search_results
from src.services.cultural_fit_service import get_cultural_fit_service
from src.services.cv_parser_service import get_cv_parser_service
from src.api.schemas.career import CVUploadResponse, CareerProfile
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# CV upload constraints
ALLOWED_CV_EXTENSIONS = ['.pdf', '.docx', '.doc']
MAX_CV_SIZE = 10 * 1024 * 1024  # 10 MB


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's message")
    ocean_scores: Dict[str, float] = Field(
        ...,
        description="OCEAN scores (openness, conscientiousness, extraversion, agreeableness, neuroticism)"
    )
    derived_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional derived personality metrics"
    )
    interpretations: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional trait interpretations with descriptions and insights"
    )
    user_transcript: Optional[str] = Field(
        None,
        description="User's spoken responses from the video recording"
    )
    message_history: Optional[List[ChatMessage]] = Field(
        None,
        description="Optional previous messages for context"
    )
    career_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="Career profile from uploaded CV for enhanced job matching"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "How can I improve my conscientiousness?",
                "ocean_scores": {
                    "openness": 0.72,
                    "conscientiousness": 0.45,
                    "extraversion": 0.58,
                    "agreeableness": 0.65,
                    "neuroticism": 0.38
                },
                "message_history": [
                    {"role": "user", "content": "What are my personality strengths?"},
                    {"role": "assistant", "content": "Based on your profile..."}
                ]
            }
        }


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str = Field(..., description="AI coach's response")
    success: bool = Field(True, description="Whether the request was successful")


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/", response_model=ChatResponse)
async def chat_with_coach(request: ChatRequest):
    """
    Send a message to the AI personality coach.

    The coach uses the user's OCEAN personality scores to provide
    personalized advice and insights.

    Returns:
        ChatResponse with the coach's reply
    """
    if not settings.AI_REPORT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI coaching is not enabled. Set AI_REPORT_ENABLED=true in configuration."
        )

    try:
        coach = get_personality_coach()

        # Convert message history to the expected format
        history = None
        if request.message_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.message_history
            ]

        response = await coach.chat(
            message=request.message,
            ocean_scores=request.ocean_scores,
            derived_metrics=request.derived_metrics,
            interpretations=request.interpretations,
            user_transcript=request.user_transcript,
            message_history=history,
            career_profile=request.career_profile
        )

        return ChatResponse(response=response, success=True)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get response from coach: {str(e)}"
        )


@router.post("/stream")
async def chat_with_coach_stream(request: ChatRequest):
    """
    Send a message to the AI personality coach with streaming response.

    Uses Server-Sent Events (SSE) to stream the response in real-time.
    Each chunk is sent as a JSON object with the text delta.

    Event types:
    - data: {"type": "chunk", "content": "..."} - Text chunk
    - data: {"type": "done"} - Stream complete
    - data: {"type": "error", "message": "..."} - Error occurred

    Returns:
        StreamingResponse with SSE events
    """
    if not settings.AI_REPORT_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI coaching is not enabled. Set AI_REPORT_ENABLED=true in configuration."
        )

    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate SSE events from the agent stream."""
        try:
            coach = get_personality_coach()

            # Convert message history to the expected format
            history = None
            if request.message_history:
                history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.message_history
                ]

            # Stream the response
            async for chunk in coach.chat_stream(
                message=request.message,
                ocean_scores=request.ocean_scores,
                derived_metrics=request.derived_metrics,
                interpretations=request.interpretations,
                user_transcript=request.user_transcript,
                message_history=history,
                career_profile=request.career_profile
            ):
                # Send chunk as SSE event
                event_data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {event_data}\n\n"

            # Check if job results are available (don't send data, just signal)
            # Frontend will fetch via separate HTTP request to avoid SSE fragmentation
            # Use peek to check without clearing - actual fetch will clear
            job_results = peek_job_search_results()
            has_jobs = job_results and len(job_results.get('jobs', [])) > 0

            # Send completion event with jobs_available flag
            done_event = {"type": "done", "jobs_available": has_jobs}
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            error_data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/job-results")
async def get_job_results():
    """
    Get the last job search results.

    Called by frontend after streaming completes to fetch job results
    via a separate HTTP request (avoids SSE fragmentation issues).

    Returns:
        Job results if available, or empty object
    """
    job_results = get_last_job_search_results()
    if job_results:
        return job_results
    return {"jobs": []}


@router.get("/health")
async def chat_health():
    """
    Check if the chat service is available.

    Returns:
        Status of the chat service
    """
    return {
        "status": "available" if settings.AI_REPORT_ENABLED else "disabled",
        "provider": settings.AI_PROVIDER if settings.AI_REPORT_ENABLED else None
    }


# =============================================================================
# Cultural Fit Endpoint
# =============================================================================

class CulturalFitRequest(BaseModel):
    """Request body for cultural fit endpoint."""
    ocean_scores: Dict[str, float] = Field(
        ...,
        description="OCEAN scores (openness, conscientiousness, extraversion, agreeableness, neuroticism)"
    )
    derived_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional derived personality metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ocean_scores": {
                    "openness": 0.72,
                    "conscientiousness": 0.45,
                    "extraversion": 0.58,
                    "agreeableness": 0.65,
                    "neuroticism": 0.38
                }
            }
        }


@router.post("/cultural-fit")
async def get_cultural_fit(request: CulturalFitRequest):
    """
    Get cultural fit analysis matching personality to workplace culture types.

    Analyzes OCEAN scores to compute 8 culture dimensions and matches
    against 12 workplace culture types using cosine similarity.

    Returns:
        Cultural fit summary with top matches, dimensions, and recommendations
    """
    try:
        service = get_cultural_fit_service()
        result = service.get_cultural_fit_summary(
            ocean_scores=request.ocean_scores,
            derived_metrics=request.derived_metrics
        )

        return result

    except Exception as e:
        logger.error(f"Cultural fit error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute cultural fit: {str(e)}"
        )


# =============================================================================
# CV Upload Endpoint
# =============================================================================

@router.post("/upload-cv", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile = File(..., description="CV file (PDF or DOCX)")
):
    """
    Upload a CV and extract career information.

    The extracted profile can be passed with chat requests to enhance
    job search and career coaching advice.

    Supported formats: PDF, DOCX, DOC
    Max file size: 10MB

    Returns:
        CVUploadResponse with extracted CareerProfile
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_CV_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_CV_EXTENSIONS)}"
        )

    # Read content
    content = await file.read()

    # Check size
    if len(content) > MAX_CV_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) // 1024 // 1024}MB). Maximum size: {MAX_CV_SIZE // 1024 // 1024}MB"
        )

    try:
        parser = get_cv_parser_service()
        profile = await parser.parse_cv(content, file.filename)

        logger.info(f"Successfully parsed CV: {file.filename} -> role={profile.current_role}, location={profile.location}")

        return CVUploadResponse(
            success=True,
            career_profile=profile
        )

    except ValueError as e:
        logger.warning(f"CV parsing failed: {e}")
        return CVUploadResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"CV upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CV: {str(e)}"
        )
