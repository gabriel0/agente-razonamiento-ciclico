import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "data" / "checkpoints.sqlite"
TRAZA_PATH = ROOT_DIR / "trazas" / "ejecucion.json"

RECURSION_LIMIT = 10
THREAD_ID = "sesion-cliente-102"

load_dotenv(ROOT_DIR / ".env", override=True)


def build_chat_model() -> BaseChatModel:
    """Ollama por defecto. Gemini solo si LLM_PROVIDER=google."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    temperature = float(os.getenv("TEMPERATURE", "0"))

    if provider in {"ollama", "local"}:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=temperature,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY para el proveedor 'google'")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
        )

    raise ValueError(
        f"Proveedor no soportado: '{provider}'. Usá 'ollama' o 'google' (Gemini)."
    )


def nombre_modelo() -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "google":
        return os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    return os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
