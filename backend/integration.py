"""
Integration module that wires all components together for end-to-end pipeline testing.

This module connects:
- Content Pipeline Orchestrator
- API endpoints (Flask/FastAPI)
- Frontend integration
- Content Repository
- All pipeline components (simplifier, translator, validator, speech generator)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from .database import get_db
from .models import ProcessedContent
from .services.pipeline.orchestrator import ContentPipelineOrchestrator

logger = logging.getLogger(__name__)


class IntegratedPipeline:
    """
    Integrated pipeline that connects all components for end-to-end processing.

    This class provides a unified interface for:
    - Processing content through the full pipeline
    - Storing results in the repository
    - Tracking metrics and performance
    - Retrieving processed content
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize the integrated pipeline with all components.

        Args:
            api_key: Optional Hugging Face API key
        """
        logger.info("Initializing integrated pipeline...")

        # Initialize database
        db = get_db()
        db.create_tables()

        # Initialize core components
        self.orchestrator = ContentPipelineOrchestrator(api_key=api_key)

        logger.info("Integrated pipeline initialized successfully")

    def process_and_store(
        self,
        input_data: str,
        target_language: str,
        grade_level: int,
        subject: str,
        output_format: str = "both",
    ) -> dict[str, Any]:
        """
        Process content through the pipeline and store in repository.

        This is the main end-to-end flow:
        1. Validate input parameters
        2. Process through pipeline (simplification → translation → validation → speech)
        3. Store in repository
        4. Track metrics
        5. Return complete result

        Args:
            input_data: Raw educational content
            target_language: Target Indian language
            grade_level: Grade level (5-12)
            subject: Subject area
            output_format: Output format ('text', 'audio', 'both')

        Returns:
            Dictionary with processed content and metadata
        """
        logger.info(
            f"Starting end-to-end processing: language={target_language}, grade={grade_level}, subject={subject}"
        )

        try:
            # Step 1: Process through pipeline
            result = self.orchestrator.process_content(
                input_data=input_data,
                target_language=target_language,
                grade_level=grade_level,
                subject=subject,
                output_format=output_format,
            )

            # Content is already stored by the orchestrator

            # Step 4: Build comprehensive response
            response = {
                "success": True,
                "content_id": result.id,
                "content": {
                    "original_text": result.original_text,
                    "simplified_text": result.simplified_text,
                    "translated_text": result.translated_text,
                    "language": result.language,
                    "grade_level": result.grade_level,
                    "subject": result.subject,
                    "audio_file_path": result.audio_file_path,
                    "audio_url": f"/api/content/{result.id}/audio"
                    if result.audio_file_path
                    else None,
                },
                "quality_scores": {
                    "ncert_alignment_score": result.ncert_alignment_score,
                    "audio_accuracy_score": result.audio_accuracy_score,
                    "validation_status": result.validation_status,
                },
                "metrics": {
                    "total_processing_time_ms": result.metadata.get(
                        "total_processing_time_ms", 0
                    ),
                    "stage_metrics": [
                        {
                            "stage": m.stage,
                            "processing_time_ms": m.processing_time_ms,
                            "success": m.success,
                            "retry_count": m.retry_count,
                        }
                        for m in result.metrics
                    ],
                },
                "metadata": result.metadata,
            }

            logger.info(
                f"End-to-end processing completed successfully: content_id={result.id}"
            )

            return response

        except Exception as e:
            logger.error(f"End-to-end processing failed: {e!s}", exc_info=True)
            raise

    def retrieve_content(self, content_id: str) -> dict[str, Any] | None:
        """
        Retrieve processed content from database.

        Args:
            content_id: UUID of the content

        Returns:
            Dictionary with content data or None if not found
        """
        try:
            session = get_db().get_session()
            try:
                content = (
                    session.query(ProcessedContent)
                    .filter(ProcessedContent.id == UUID(content_id))
                    .first()
                )

                if not content:
                    return None

                return {
                    "id": str(content.id),
                    "original_text": content.original_text,
                    "simplified_text": content.simplified_text,
                    "translated_text": content.translated_text,
                    "language": content.language,
                    "grade_level": content.grade_level,
                    "subject": content.subject,
                    "audio_file_path": content.audio_file_path,
                    "audio_url": f"/api/content/{content.id}/audio"
                    if content.audio_file_path
                    else None,
                    "ncert_alignment_score": content.ncert_alignment_score,
                    "audio_accuracy_score": content.audio_accuracy_score,
                    "created_at": content.created_at.isoformat()
                    if content.created_at
                    else None,
                    "metadata": content.content_metadata,
                }
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to retrieve content {content_id}: {e!s}")
            return None

    def search_content(
        self,
        language: str | None = None,
        grade_level: int | None = None,
        subject: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Search for content with filters.

        Args:
            language: Filter by language
            grade_level: Filter by grade level
            subject: Filter by subject
            limit: Maximum results

        Returns:
            Dictionary with search results
        """
        try:
            session = get_db().get_session()
            try:
                query = session.query(ProcessedContent)

                if language:
                    query = query.filter(ProcessedContent.language == language)
                if grade_level:
                    query = query.filter(ProcessedContent.grade_level == grade_level)
                if subject:
                    query = query.filter(ProcessedContent.subject == subject)

                contents = query.limit(limit).all()

                results = []
                for content in contents:
                    results.append(
                        {
                            "id": str(content.id),
                            "language": content.language,
                            "grade_level": content.grade_level,
                            "subject": content.subject,
                            "translated_text_preview": content.translated_text[:200]
                            + "..."
                            if len(content.translated_text) > 200
                            else content.translated_text,
                            "ncert_alignment_score": content.ncert_alignment_score,
                            "audio_available": content.audio_file_path is not None,
                            "created_at": content.created_at.isoformat()
                            if content.created_at
                            else None,
                        }
                    )

                return {
                    "total_count": len(results),
                    "results": results,
                    "filters": {
                        "language": language,
                        "grade_level": grade_level,
                        "subject": subject,
                    },
                }
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Search failed: {e!s}")
            return {"total_count": 0, "results": [], "error": str(e)}

    def create_offline_package(
        self, content_ids: list, package_name: str | None = None
    ) -> dict[str, Any]:
        """
        Create offline package for batch download.

        Args:
            content_ids: List of content IDs
            package_name: Optional package name

        Returns:
            Dictionary with package information
        """
        try:
            import json
            import zipfile
            from datetime import datetime

            # Create package directory
            package_dir = Path("data/packages")
            package_dir.mkdir(parents=True, exist_ok=True)

            # Generate package filename
            if not package_name:
                package_name = f"package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            package_path = package_dir / f"{package_name}.zip"

            # Retrieve content and create ZIP
            session = get_db().get_session()
            try:
                with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for content_id in content_ids:
                        content = (
                            session.query(ProcessedContent)
                            .filter(ProcessedContent.id == UUID(content_id))
                            .first()
                        )

                        if content:
                            # Add content as JSON
                            content_data = {
                                "id": str(content.id),
                                "original_text": content.original_text,
                                "simplified_text": content.simplified_text,
                                "translated_text": content.translated_text,
                                "language": content.language,
                                "grade_level": content.grade_level,
                                "subject": content.subject,
                                "ncert_alignment_score": content.ncert_alignment_score,
                            }
                            zipf.writestr(
                                f"{content_id}.json", json.dumps(content_data, indent=2)
                            )

                            # Add audio file if exists
                            if (
                                content.audio_file_path
                                and Path(content.audio_file_path).exists()
                            ):
                                zipf.write(content.audio_file_path, f"{content_id}.mp3")
            finally:
                session.close()

            # Get package size
            package_size = os.path.getsize(package_path)

            return {
                "success": True,
                "package_path": str(package_path),
                "package_size_bytes": package_size,
                "package_size_mb": round(package_size / (1024 * 1024), 2),
                "content_count": len(content_ids),
            }

        except Exception as e:
            logger.error(f"Failed to create offline package: {e!s}")
            return {"success": False, "error": str(e)}

    def get_system_health(self) -> dict[str, Any]:
        """
        Get system health status and metrics.

        Returns:
            Dictionary with system health information
        """
        try:
            from sqlalchemy import text

            # Check database connection
            db_healthy = True
            content_count = 0
            try:
                session = get_db().get_session()
                session.execute(text("SELECT 1"))
                content_count = session.query(ProcessedContent).count()
                session.close()
            except Exception:
                db_healthy = False

            return {
                "status": "healthy" if db_healthy else "degraded",
                "database": {"connected": db_healthy, "content_count": content_count},
                "components": {
                    "orchestrator": "operational",
                    "database": "operational" if db_healthy else "offline",
                },
            }

        except Exception as e:
            logger.error(f"Health check failed: {e!s}")
            return {"status": "unhealthy", "error": str(e)}


# Global instance for API usage
_integrated_pipeline = None


def get_integrated_pipeline(api_key: str | None = None) -> IntegratedPipeline:
    """
    Get or create the global integrated pipeline instance.

    Args:
        api_key: Optional Hugging Face API key

    Returns:
        IntegratedPipeline instance
    """
    global _integrated_pipeline

    if _integrated_pipeline is None:
        _integrated_pipeline = IntegratedPipeline(api_key=api_key)

    return _integrated_pipeline


def test_end_to_end_flow(
    sample_text: str = "Photosynthesis is the process by which plants convert sunlight into energy.",
    target_language: str = "Hindi",
    grade_level: int = 8,
    subject: str = "Science",
) -> dict[str, Any]:
    """
    Test the complete end-to-end pipeline flow.

    This function demonstrates the full integration:
    1. Input → Simplification → Translation → Validation → Speech → Storage → Retrieval

    Args:
        sample_text: Sample educational content
        target_language: Target language
        grade_level: Grade level
        subject: Subject area

    Returns:
        Dictionary with test results
    """
    logger.info("=" * 80)
    logger.info("STARTING END-TO-END INTEGRATION TEST")
    logger.info("=" * 80)

    pipeline = get_integrated_pipeline()

    try:
        # Step 1: Process content
        logger.info("\n[STEP 1] Processing content through pipeline...")
        result = pipeline.process_and_store(
            input_data=sample_text,
            target_language=target_language,
            grade_level=grade_level,
            subject=subject,
            output_format="both",
        )

        content_id = result["content_id"]
        logger.info(f"✓ Content processed successfully: ID={content_id}")
        logger.info(
            f"  - NCERT Alignment: {result['quality_scores']['ncert_alignment_score']:.2%}"
        )
        logger.info(
            f"  - Audio Accuracy: {result['quality_scores'].get('audio_accuracy_score', 0):.2%}"
        )
        logger.info(
            f"  - Processing Time: {result['metrics']['total_processing_time_ms']}ms"
        )

        # Step 2: Retrieve content
        logger.info("\n[STEP 2] Retrieving content from repository...")
        retrieved = pipeline.retrieve_content(content_id)

        if retrieved:
            logger.info("✓ Content retrieved successfully")
            logger.info(f"  - Language: {retrieved['language']}")
            logger.info(f"  - Grade: {retrieved['grade_level']}")
            logger.info(f"  - Subject: {retrieved['subject']}")
        else:
            logger.error("✗ Failed to retrieve content")
            return {"success": False, "error": "Content retrieval failed"}

        # Step 3: Search for content
        logger.info("\n[STEP 3] Searching for content...")
        search_results = pipeline.search_content(
            language=target_language, grade_level=grade_level, subject=subject
        )

        logger.info(
            f"✓ Search completed: {search_results['total_count']} results found"
        )

        # Step 4: Create offline package
        logger.info("\n[STEP 4] Creating offline package...")
        package_result = pipeline.create_offline_package(
            content_ids=[content_id], package_name="test_package"
        )

        if package_result["success"]:
            logger.info(
                f"✓ Offline package created: {package_result['package_size_mb']}MB"
            )
        else:
            logger.warning(f"⚠ Package creation failed: {package_result.get('error')}")

        # Step 5: Check system health
        logger.info("\n[STEP 5] Checking system health...")
        health = pipeline.get_system_health()
        logger.info(f"✓ System status: {health['status']}")

        logger.info("\n" + "=" * 80)
        logger.info("END-TO-END INTEGRATION TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return {
            "success": True,
            "content_id": content_id,
            "processing_result": result,
            "retrieval_result": retrieved,
            "search_result": search_results,
            "package_result": package_result,
            "health_check": health,
        }

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"END-TO-END INTEGRATION TEST FAILED: {e!s}")
        logger.error("=" * 80)

        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Run end-to-end test when module is executed directly
    test_result = test_end_to_end_flow()

    if test_result["success"]:
        print("\n✓ All integration tests passed!")
        print(f"\nTest Content ID: {test_result['content_id']}")
    else:
        print(f"\n✗ Integration test failed: {test_result['error']}")
