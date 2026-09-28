# Gen AI – Assignment 1: FastAPI with Bigram Model and spaCy Word Embeddings

A FastAPI application built on the Module 3 class activity (bigram text generation), extended with word-embedding endpoints using spaCy (`en_core_web_lg`, 300-dimensional vectors).

## Project structure

```
app/
├── __init__.py
├── bigram_model.py   # Bigram language model (text generation)
└── main.py           # FastAPI app and endpoints
Dockerfile
pyproject.toml
uv.lock
```

## Endpoints

| Method | Endpoint      | Body example                               | Returns                                   |
|--------|---------------|--------------------------------------------|-------------------------------------------|
| GET    | `/`           | –                                          | `{"Hello": "World"}`                      |
| POST   | `/generate`   | `{"start_word": "bigram", "length": 5}`    | Text generated with the bigram model      |
| POST   | `/embedding`  | `{"word": "king"}`                         | The 300-dim spaCy embedding for the word  |
| POST   | `/similarity` | `{"word1": "king", "word2": "queen"}`      | Cosine similarity between two embeddings  |

## Run with Docker

```bash
docker build -t gen-ai .
docker run -p 8000:80 gen-ai
```

Then open http://127.0.0.1:8000/docs to try the endpoints.

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'
```