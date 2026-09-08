import os

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
