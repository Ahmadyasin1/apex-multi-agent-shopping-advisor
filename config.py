"""
Central configuration — all tunable values live here.
Never hardcode values in agent files; reference Config instead.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── LLM Settings ─────────────────────────────────────────────────────────
    LOCAL_LLM        = os.getenv("LOCAL_LLM_MODEL", "llama3")
    PREMIUM_LLM      = os.getenv("PREMIUM_LLM_MODEL", "gemini-2.5-pro")
    GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
    RAPIDAPI_KEY     = os.getenv("RAPIDAPI_KEY", "dd08720706msh701d2fc376013e2p1c207cjsn9aa9cfbed80c")

    # Ordered list of Gemini models to try (most capable → fastest)
    GEMINI_MODEL_LIST = [
        m.strip() for m in
        os.getenv("GEMINI_MODELS", "gemini-2.0-flash,gemini-2.5-pro,gemini-1.5-pro").split(",")
        if m.strip()
    ]

    # ── Embedding & Vector DB ─────────────────────────────────────────────────
    EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH  = os.getenv("FAISS_INDEX_PATH", "data_layer/products.index")
    METADATA_PATH     = os.getenv("METADATA_PATH",    "data_layer/metadata.json")
    DATASET_PATH      = os.getenv("DATASET_PATH",     "data_layer/dataset.json")

    # ── Retrieval Settings ────────────────────────────────────────────────────
    TOP_K_SIMPLE   = int(os.getenv("TOP_K_SIMPLE",  "15"))
    TOP_K_COMPLEX  = int(os.getenv("TOP_K_COMPLEX", "20"))
    MAX_RESULTS    = int(os.getenv("MAX_RESULTS",    "8"))

    # ── Critique Settings ─────────────────────────────────────────────────────
    MAX_CRITIQUE_REVISIONS = int(os.getenv("MAX_CRITIQUE_REVISIONS", "2"))

    # ── Category aliases for detection (pipe-separated: alias=canonical) ─────
    # Format: "alias1=canonical,alias2=canonical,..."
    CATEGORY_ALIASES_RAW = os.getenv(
        "CATEGORY_ALIASES",
        "headphones=headphones,earphones=headphones,earbuds=headphones,"
        "laptop=laptop,notebook=laptop,chromebook=laptop,"
        "smartphone=smartphone,iphone=smartphone,android=smartphone,"
        "smartwatch=smartwatch,watch=smartwatch,"
        "desktop=desktop,imac=desktop,"
        "tablet=tablet,ipad=tablet,"
        "camera=camera,mirrorless=camera,dslr=camera"
    )

    @classmethod
    def get_category_aliases(cls) -> list:
        """Returns [(alias, canonical), ...] parsed from env var."""
        result = []
        for pair in cls.CATEGORY_ALIASES_RAW.split(","):
            pair = pair.strip()
            if "=" in pair:
                alias, canonical = pair.split("=", 1)
                result.append((alias.strip(), canonical.strip()))
        return result

    # ── Scoring weights (can be env-overridden) ───────────────────────────────
    DEFAULT_PRIORITY_VECTOR = {
        "price":       float(os.getenv("DEFAULT_PRICE_WEIGHT",       "0.20")),
        "performance": float(os.getenv("DEFAULT_PERFORMANCE_WEIGHT", "0.25")),
        "brand":       float(os.getenv("DEFAULT_BRAND_WEIGHT",       "0.20")),
        "durability":  float(os.getenv("DEFAULT_DURABILITY_WEIGHT",  "0.15")),
        "innovation":  float(os.getenv("DEFAULT_INNOVATION_WEIGHT",  "0.20")),
    }

    # ── App Settings ──────────────────────────────────────────────────────────
    APP_TITLE    = os.getenv("APP_TITLE",    "APEX — Intelligent Shopping Advisor")
    APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Elite Multi-Agent AI Powered by LangGraph")
    AUTHOR       = os.getenv("AUTHOR",       "Ahmad Yasin")
