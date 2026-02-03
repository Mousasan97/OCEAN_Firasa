"""
Career Profile schemas for CV upload feature.

Structured data extracted from user CVs for enhanced job matching.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class CareerProfile(BaseModel):
    """Structured career information extracted from CV."""
    current_role: Optional[str] = Field(None, description="Current or most recent job title")
    target_role: Optional[str] = Field(None, description="Desired job role if mentioned")
    location: Optional[str] = Field(None, description="Current location or preferred location")
    years_experience: Optional[int] = Field(None, description="Total years of work experience")
    key_skills: List[str] = Field(default_factory=list, description="Top technical and soft skills (max 10)")
    industries: List[str] = Field(default_factory=list, description="Industries worked in")
    education_level: Optional[str] = Field(None, description="Highest education level")
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")
    summary: Optional[str] = Field(None, description="Brief professional summary (2-3 sentences)")


class CVUploadResponse(BaseModel):
    """Response from CV upload endpoint."""
    success: bool = Field(..., description="Whether CV was parsed successfully")
    career_profile: Optional[CareerProfile] = Field(None, description="Extracted career profile")
    error: Optional[str] = Field(None, description="Error message if parsing failed")
