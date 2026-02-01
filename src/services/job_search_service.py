"""
Job Search Service using SerpAPI Google Jobs

Fetches job listings based on role and location, then provides data
for AI-based cultural fit analysis.
"""
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import httpx

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JobListing:
    """Represents a job listing from Google Jobs."""
    title: str
    company_name: str
    location: str
    description: str
    qualifications: List[str]
    posted_at: str
    employment_type: str
    job_id: str
    apply_link: Optional[str] = None
    salary: Optional[str] = None
    company_logo: Optional[str] = None
    extensions: Optional[List[str]] = None


class JobSearchService:
    """Service for searching jobs using SerpAPI Google Jobs."""

    def __init__(self):
        self.api_key = settings.SERPAPI_KEY
        self.base_url = "https://serpapi.com/search"

    def search_jobs(
        self,
        query: str,
        location: str,
        num_results: int = 10
    ) -> List[JobListing]:
        """
        Search for jobs using Google Jobs via SerpAPI.

        Args:
            query: Job title or role (e.g., "software engineer", "data scientist")
            location: Location (e.g., "Milan, Italy", "New York, USA")
            num_results: Maximum number of results to return

        Returns:
            List of JobListing objects
        """
        if not self.api_key:
            logger.error("SERPAPI_KEY not configured")
            raise ValueError("Job search is not configured. Please set SERPAPI_KEY in environment.")

        params = {
            "api_key": self.api_key,
            "engine": "google_jobs",
            "q": query,
            "location": location,
            "num": num_results
        }

        try:
            logger.info(f"Searching jobs: query='{query}', location='{location}'")

            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()

            jobs_results = data.get("jobs_results", [])
            logger.info(f"Found {len(jobs_results)} jobs")

            jobs = []
            for job_data in jobs_results[:num_results]:
                job = self._parse_job(job_data)
                if job:
                    jobs.append(job)

            return jobs

        except httpx.HTTPError as e:
            logger.error(f"HTTP error searching jobs: {e}")
            raise
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            raise

    def _parse_job(self, data: Dict[str, Any]) -> Optional[JobListing]:
        """Parse raw job data into JobListing object."""
        try:
            # Extract qualifications from highlights if available
            qualifications = []
            highlights = data.get("job_highlights", [])
            for highlight in highlights:
                if highlight.get("title", "").lower() in ["qualifications", "requirements"]:
                    qualifications.extend(highlight.get("items", []))

            # Extract employment type from detected extensions
            extensions = data.get("detected_extensions", {})
            employment_type = extensions.get("schedule_type", "")
            posted_at = extensions.get("posted_at", "")
            salary = extensions.get("salary", "")

            return JobListing(
                title=data.get("title", "Unknown Title"),
                company_name=data.get("company_name", "Unknown Company"),
                location=data.get("location", ""),
                description=data.get("description", ""),
                qualifications=qualifications,
                posted_at=posted_at,
                employment_type=employment_type,
                job_id=data.get("job_id", ""),
                apply_link=data.get("apply_options", [{}])[0].get("link") if data.get("apply_options") else None,
                salary=salary if salary else None,
                company_logo=data.get("thumbnail"),
                extensions=data.get("extensions", [])
            )
        except Exception as e:
            logger.warning(f"Error parsing job data: {e}")
            return None

    def format_job_for_analysis(self, job: JobListing) -> str:
        """Format a job listing for AI cultural analysis."""
        parts = [
            f"**{job.title}** at **{job.company_name}**",
            f"Location: {job.location}",
        ]

        if job.employment_type:
            parts.append(f"Type: {job.employment_type}")

        if job.salary:
            parts.append(f"Salary: {job.salary}")

        if job.posted_at:
            parts.append(f"Posted: {job.posted_at}")

        parts.append(f"\nDescription:\n{job.description[:1000]}...")

        if job.qualifications:
            parts.append(f"\nQualifications:\n- " + "\n- ".join(job.qualifications[:5]))

        return "\n".join(parts)

    def jobs_to_dict(self, jobs: List[JobListing]) -> List[Dict[str, Any]]:
        """Convert job listings to dictionaries for JSON serialization."""
        return [
            {
                "title": job.title,
                "company_name": job.company_name,
                "location": job.location,
                "description": job.description,
                "qualifications": job.qualifications,
                "posted_at": job.posted_at,
                "employment_type": job.employment_type,
                "job_id": job.job_id,
                "apply_link": job.apply_link,
                "salary": job.salary,
                "company_logo": job.company_logo,
                "extensions": job.extensions
            }
            for job in jobs
        ]


# Singleton instance
_job_search_service: Optional[JobSearchService] = None


def get_job_search_service() -> JobSearchService:
    """Get or create singleton job search service."""
    global _job_search_service
    if _job_search_service is None:
        _job_search_service = JobSearchService()
    return _job_search_service


def search_jobs_for_agent(
    role: str,
    location: str,
    num_results: int = 8
) -> str:
    """
    Search for jobs and format results for AI agent consumption.

    Args:
        role: Job role/title to search for
        location: Location to search in
        num_results: Number of results to return

    Returns:
        Formatted string with job listings for AI analysis
    """
    service = get_job_search_service()

    try:
        jobs = service.search_jobs(role, location, num_results)

        if not jobs:
            return f"No jobs found for '{role}' in '{location}'. Try a different search term or location."

        # Format for AI to analyze
        result_parts = [
            f"Found {len(jobs)} jobs for **{role}** in **{location}**:\n",
            "---"
        ]

        for i, job in enumerate(jobs, 1):
            result_parts.append(f"\n### Job {i}: {job.title}")
            result_parts.append(f"**Company:** {job.company_name}")
            result_parts.append(f"**Location:** {job.location}")

            if job.employment_type:
                result_parts.append(f"**Type:** {job.employment_type}")
            if job.salary:
                result_parts.append(f"**Salary:** {job.salary}")
            if job.posted_at:
                result_parts.append(f"**Posted:** {job.posted_at}")

            # Truncate description for analysis
            desc = job.description[:800] + "..." if len(job.description) > 800 else job.description
            result_parts.append(f"\n**Description:**\n{desc}")

            if job.qualifications:
                quals = job.qualifications[:5]
                result_parts.append(f"\n**Key Requirements:**")
                for qual in quals:
                    result_parts.append(f"- {qual}")

            result_parts.append("\n---")

        return "\n".join(result_parts)

    except Exception as e:
        logger.error(f"Error in search_jobs_for_agent: {e}")
        return f"Error searching for jobs: {str(e)}"
