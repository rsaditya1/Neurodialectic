from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import logging
import traceback
import os

# Import everything from your backend
from backend import create_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuraDialectic API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    max_iterations: Optional[int] = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    final_answer: Optional[str] = None
    summary: Optional[str] = None
    generator_output: Optional[str] = None
    critic_output: Optional[str] = None
    validator_output: Optional[str] = None
    refinement_outputs: list = []
    confidence: float = 0.0
    iterations_used: int = 0


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NeuraDialectic"}


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query[:100]}...")

        workflow = create_workflow(request.query, request.max_iterations)
        compiled = workflow.compile()

        result = await asyncio.to_thread(
            compiled.invoke,
            {"query": request.query}
        )

        response = QueryResponse(
            final_answer=result.get("final_answer"),
            summary=result.get("summary"),
            generator_output=result.get("generator_output"),
            critic_output=result.get("critic_output"),
            validator_output=result.get("validator_output"),
            refinement_outputs=result.get("refinement_outputs", []),
            confidence=result.get("confidence", 0.0),
            iterations_used=result.get("iteration", 0)
        )

        logger.info(f"Query completed. Confidence: {response.confidence}")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


# Serve the frontend HTML at the root URL
@app.get("/")
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(html_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)