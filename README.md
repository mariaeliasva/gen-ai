# Gen AI – FastAPI Project

FastAPI application for the Applied Generative AI course (Columbia SPS). It builds on the Module 3 class activity (bigram text generation) and adds new endpoints with each assignment:

- **Assignment 1:** word embeddings with spaCy (`en_core_web_lg`, 300-dimensional vectors)
- **Assignment 2:** CNN image classifier trained on CIFAR-10
- **Assignment 3:** GAN that generates hand-written digits, trained on MNIST

## Project structure

```
app/
├── __init__.py
├── bigram_model.py    # Bigram language model (text generation)
├── cnn_model.py       # Assignment 2 CNN architecture
├── cnn_weights.pth    # Trained CNN weights
├── gan_model.py       # Assignment 3 GAN (generator + discriminator)
├── gan_generator.pth  # Trained GAN generator weights
└── main.py            # FastAPI app and endpoints
train_cnn.py           # Trains the CNN on CIFAR-10 and saves the weights
train_gan.py           # Trains the GAN on MNIST and saves the generator
Dockerfile
pyproject.toml
uv.lock
```

## Endpoints

| Method | Endpoint      | Input                                   | Returns                                        |
|--------|---------------|-----------------------------------------|------------------------------------------------|
| GET    | `/`           | –                                       | `{"Hello": "World"}`                           |
| POST   | `/generate`   | `{"start_word": "bigram", "length": 5}` | Text generated with the bigram model           |
| POST   | `/embedding`  | `{"word": "king"}`                      | 300-dim spaCy embedding for the word           |
| POST   | `/similarity` | `{"word1": "king", "word2": "queen"}`   | Cosine similarity between two embeddings       |
| POST   | `/classify`   | Image file upload (form field `file`)   | Predicted CIFAR-10 class + top-3 probabilities |
| GET    | `/generate-digits` | Query params `num_images` (1–64), optional `seed` | PNG grid of GAN-generated digits |

## Assignment 2: CNN image classifier

`app/cnn_model.py` implements the required architecture:

```
Input: RGB image 64x64x3
Conv2D(16 filters, 3x3, stride 1, padding 1) → ReLU → MaxPool(2x2, stride 2)
Conv2D(32 filters, 3x3, stride 1, padding 1) → ReLU → MaxPool(2x2, stride 2)
Flatten → Fully connected (100) → ReLU → Fully connected (10)
```

CIFAR-10 images (32x32) are resized to 64x64 to match the input size. Trained for 5 epochs with Adam (lr = 0.001): **test accuracy 65.27%**.

To retrain the model and regenerate `app/cnn_weights.pth`:

```bash
uv run python train_cnn.py
```

## Assignment 3: GAN for hand-written digits

`app/gan_model.py` implements the required architecture:

```
Generator:     noise (100) → Linear → 7x7x128 → ConvTranspose2d(128→64, k4 s2 p1) → BatchNorm → ReLU
               → ConvTranspose2d(64→1, k4 s2 p1) → Tanh → 28x28 image
Discriminator: image (1x28x28) → Conv2d(1→64, k4 s2 p1) → LeakyReLU(0.2)
               → Conv2d(64→128, k4 s2 p1) → BatchNorm → LeakyReLU(0.2) → Flatten → Linear → Sigmoid
```

Trained on MNIST for 20 epochs with Adam (lr = 0.0002, betas = (0.5, 0.999)) and binary cross-entropy loss. Only the generator is saved, since the API only needs it to create new digits.

To retrain and regenerate `app/gan_generator.pth`:

```bash
uv run python train_gan.py
```

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

## Example requests

```bash
# Word embedding
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "king"}'

# Image classification
curl -X POST http://127.0.0.1:8000/classify \
  -F "file=@airplane.jpg"
```