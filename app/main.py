import io
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import spacy
import torch

from app.bigram_model import BigramModel
from app.cnn_model import CNN, TRANSFORM, CIFAR10_CLASSES

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


# ---------------------------------------------------------------
# Assignment 2: CNN image classifier (CIFAR-10)
# ---------------------------------------------------------------
WEIGHTS_PATH = Path(__file__).parent / "cnn_weights.pth"

cnn_model = CNN()
cnn_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
cnn_model.eval()  # inference mode


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image and get its predicted CIFAR-10 class."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image")

    x = TRANSFORM(image).unsqueeze(0)  # (1, 3, 64, 64)
    with torch.no_grad():
        probabilities = torch.softmax(cnn_model(x), dim=1)[0]

    top_probs, top_indices = probabilities.topk(3)
    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[top_indices[0]],
        "confidence": round(float(top_probs[0]), 4),
        "top_3": [
            {"class": CIFAR10_CLASSES[i], "probability": round(float(p), 4)}
            for p, i in zip(top_probs, top_indices)
        ],
    }


# ---------------------------------------------------------------
# Assignment 3: GAN that generates hand-written digits (MNIST)
# ---------------------------------------------------------------
from fastapi import Query
from fastapi.responses import Response
from torchvision.utils import make_grid

from app.gan_model import Generator, LATENT_DIM

GAN_WEIGHTS_PATH = Path(__file__).parent / "gan_generator.pth"

gan_generator = Generator()
gan_generator.load_state_dict(torch.load(GAN_WEIGHTS_PATH, map_location="cpu"))
gan_generator.eval()  # inference mode


@app.get("/generate-digits", responses={200: {"content": {"image/png": {}}}})
def generate_digits(
    num_images: int = Query(16, ge=1, le=64, description="How many digits to generate"),
    seed: int | None = Query(None, description="Optional seed to get the same images again"),
):
    """Generate new hand-written digits with the GAN and return them as a PNG grid."""
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn(num_images, LATENT_DIM)
    with torch.no_grad():
        images = gan_generator(noise)  # (num_images, 1, 28, 28), values in [-1, 1]

    images = (images + 1) / 2  # rescale to [0, 1]
    grid = make_grid(images, nrow=min(num_images, 8), padding=2)
    array = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")
    png = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)

    buffer = io.BytesIO()
    png.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")