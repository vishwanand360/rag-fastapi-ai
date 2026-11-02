from backend.gemini_errors import gemini_error_message, invalid_gemini_credentials


def test_identifies_invalid_key_through_wrapped_sdk_error_without_echoing_secret():
    sdk_error = RuntimeError("400 INVALID_ARGUMENT: API_KEY_INVALID; API key not valid")
    wrapped = RuntimeError("Google request failed")
    wrapped.__cause__ = sdk_error

    assert invalid_gemini_credentials(wrapped)
    message = gemini_error_message(wrapped)
    assert "GEMINI_API_KEY" in message
    assert "never logged or returned" in message
    assert "not-a-real-secret" not in message


def test_does_not_call_quota_or_model_errors_invalid_credentials():
    assert not invalid_gemini_credentials(RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"))
