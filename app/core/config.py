from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "BUPT Campus Smart Life Assistant Agent"
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_DB_DIR: str = "./chroma_data"
    
    class Config:
        env_file = ".env"

settings = Settings()
