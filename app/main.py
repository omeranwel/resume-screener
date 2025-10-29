from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Pydantic model for request validation
class ResumeData(BaseModel):
    name: str
    resume: str
    job_description: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Screener API!"}

@app.post("/screen_resume/")
def screen_resume(data: ResumeData):
    # Extract resume and job description from the request
    resume = data.resume
    job_description = data.job_description

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

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
    return {"similarity_score": similarity_matrix[0][0], "category": similarity_category}