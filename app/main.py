from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

load_dotenv()

from app.api.routes import router as chat_router

app = FastAPI(
    title="BUPT Campus Smart Life Assistant Agent",
    description="北邮校园智能生活助手 API",
    version="0.1.0"
)

app.include_router(chat_router, prefix="/api")

# Mount static frontend directory
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

@app.get("/")
async def root():
    # Automatically redirect from root path to our mounted frontend index.html
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
