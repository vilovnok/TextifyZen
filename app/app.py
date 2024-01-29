from fastapi import FastAPI

from simplifier import Simplifier

simplifier = None
app = FastAPI()

@app.get("/")
def index():
    return {"text": "Text Simplifier"}


@app.on_event("startup")
def initialization():
    global simplifier
    simplifier = Simplifier()

@app.get("/simplify")
def simplify_text(text: str):
    simplified_text = simplifier.simplify(text)

    response = {
        'text': text,
        'simplified_text': simplified_text
    }

    return response