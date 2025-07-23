# DS-AI-Agent
An AI Agent that is an all-round data science assistant

## Changelog

### v0.1.1 – Bug Fixes

* **Consistent return signature for `GroqDataScienceAgent.ask`** – The method now always returns a tuple `(response, response_time)` even when the API key is missing. This prevents `ValueError: too many values to unpack (expected 2)` errors in all helper methods and UI callbacks that expect two return values.

```python
# Before
if not self.api_key:
    return "⚠️ API key not set. Please enter your Groq API key in the sidebar."

# After
if not self.api_key:
    return "⚠️ API key not set. Please enter your Groq API key in the sidebar.", 0
```

No other breaking changes were introduced.
