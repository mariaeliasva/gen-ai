from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

from app.bigram_model import BigramModel

app = FastAPI()

# ---------------------------------------------------------------
# Module 3 Class Activity: Bigram text generation
# ---------------------------------------------------------------
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# ---------------------------------------------------------------
# Assignment 1: Word embeddings with spaCy (Module 2)
# ---------------------------------------------------------------
# en_core_web_lg includes real 300-dimensional word vectors.
nlp = spacy.load("en_core_web_lg")


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    """Returns the spaCy word embedding for the query word."""
    token = nlp(request.word)[0]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for word '{request.word}'",
        )

    return {
        "word": request.word,
        "dimension": len(token.vector),
        "embedding": token.vector.tolist(),
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    """Returns the cosine similarity between two words' embeddings."""
    t1 = nlp(request.word1)[0]
    t2 = nlp(request.word2)[0]

    if not (t1.has_vector and t2.has_vector):
        raise HTTPException(status_code=404, detail="One of the words has no embedding")

    return {
        "word1": request.word1,
        "word2": request.word2,
        "similarity": float(t1.similarity(t2)),
    }