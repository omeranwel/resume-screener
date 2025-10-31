from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import asyncio

# Optional GPU: safe fallback to zero if no GPU
GPU_AVAILABLE = False
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

gpu_util_gauge = Gauge(
    "gpu_utilization_percent", "Current total GPU utilization (percent)"
)
gpu_util_gauge.set(0.0)

app = FastAPI(
    title="Resume Screener API",
    description="""
    This API compares resumes with job descriptions using TF-IDF and cosine similarity.
    It returns a similarity score and categorizes match level as High, Medium, or Low.
    """,
    version="1.0.0",
    contact={
        "name": "Omer",
        "url": "https://github.com/omeranwel/resume-screener",
        "email": "omer@example.com",
    },
)


instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)


# Pydantic model for request validation
class ResumeData(BaseModel):
    name: str
    resume: str
    job_description: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Screener API!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/screen_resume/")
def screen_resume(data: ResumeData):
    # Extract resume and job description from the request
    resume = data.resume
    job_description = data.job_description

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    # Combine the resume and job description into a single list
    documents = [resume, job_description]

    # Transform the documents into TF-IDF matrices
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Compute cosine similarity between the resume and job description
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Initialize similarity_category with a default value
    similarity_category = "Low"

    if similarity_matrix[0][0] > 0.8:
        similarity_category = "High"

    elif similarity_matrix[0][0] > 0.5:
        similarity_category = "Medium"

    else:
        similarity_category = "Low"

    # Return the similarity score as a response
    return {
        "similarity_score": similarity_matrix[0][0],
        "category": similarity_category,
    }


async def poll_gpu_forever():
    """Update GPU metric every 10s. If no GPU, keep 0."""
    while True:
        try:
            if GPU_AVAILABLE:
                # Sum utilization across all GPUs
                count = pynvml.nvmlDeviceGetCount()
                total = 0
                for i in range(count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    total += util.gpu  # percent
                avg = total / max(1, count)
                gpu_util_gauge.set(avg)
            else:
                gpu_util_gauge.set(0.0)
        except Exception:
            gpu_util_gauge.set(0.0)
        await asyncio.sleep(10)


# type: ignore[attr-defined]
@app.on_event("startup")
async def _startup():
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
    asyncio.create_task(poll_gpu_forever())


@app.on_event("shutdown")
def _shutdown():
    if GPU_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
