from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BUPT Campus Smart Life Assistant Agent"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_DB_DIR: str = "./chroma_data"
    RAG_KB_PATH: str = "./app/rag/data/campus_kb.json"
    TOOL_DATA_MODE: str = "json"  # json | sqlite | mock
    TOOL_DATA_JSON_PATH: str = "./app/tools/data/campus_data.json"
    TOOL_DATA_SQLITE_PATH: str = "./app/tools/data/campus_data.db"
    
    class Config:
        env_file = ".env"

settings = Settings()
