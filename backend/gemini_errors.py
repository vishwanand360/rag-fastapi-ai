"""Safe, actionable Gemini credential errors without echoing secrets."""


def invalid_gemini_credentials(error: Exception) -> bool:
    current: BaseException | None = error
    while current is not None:
        message = str(current).lower()
        code = str(getattr(current, "code", "")).lower()
        if (any(marker in message or marker in code for marker in (
            "api_key_invalid", "api key not valid", "invalid api key", "unauthenticated",
        )) or code in {"401", "unauthorized"}):
            return True
        current = current.__cause__ or current.__context__
    return False


def gemini_error_message(error: Exception) -> str:
    if invalid_gemini_credentials(error):
        return (
            "Gemini rejected GEMINI_API_KEY. Check that the key is valid, belongs to the intended Google project, "
            "and has the Generative Language API enabled. The key value is never logged or returned."
        )
    return "Gemini is temporarily unavailable. Check API access, quota, and model configuration."
