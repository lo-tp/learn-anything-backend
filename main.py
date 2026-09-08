import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

PORT = int(os.getenv("PORT", 8000))

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
