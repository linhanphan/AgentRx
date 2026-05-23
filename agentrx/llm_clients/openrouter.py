import os
import time
from datetime import datetime
from openai import OpenAI
import agentrx.pipeline.globals as g
try:
    import agentrx.reports.metrics as metrics
except ImportError:
    metrics = None

from agentrx.llm_clients.utils import dump_call

class LLMAgent:
    def __init__(
            self,
            api_version=None,
            model_name=None,
            model_version=None,
            deployment_name=None,
    ):
        self.api_version = api_version
        self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.model_version = model_version
        self.deployment_name = deployment_name
        
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Initialize telemetry tracking
        self.last_call_telemetry = None

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_llm_response(self, messages):
        # Capture timing before the call
        start_timestamp = datetime.now()
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        # Capture timing after the call
        end_time = time.perf_counter()
        end_timestamp = datetime.now()
        execution_time_sec = round(end_time - start_time, 4)

        # Dump raw request and response json
        request_payload = {
            "model": self.model_name,
            "messages": messages
        }
        dump_call(request_payload, response)

        # Extract token usage from response
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if hasattr(response, "usage") and response.usage is not None:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            total_tokens = getattr(response.usage, "total_tokens", 0) or 0

        # Create telemetry objects if metrics is available
        if metrics is not None:
            token_usage = metrics.TokenUsage(
                prompt_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                total_tokens=total_tokens
            )

            time_info = metrics.TimingInfo(
                start_time=start_timestamp,
                end_time=end_timestamp,
                execution_time_sec=execution_time_sec
            )

            self.last_call_telemetry = metrics.LLMCallTelemetry(
                tokens=token_usage,
                time=time_info,
                model_name=self.model_name,
                instance=self.base_url
            )

        return response

    @staticmethod
    def openrouter_mk_client() -> OpenAI:
        """Create an OpenRouter client using credentials from globals/.env."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(
            base_url=base_url,
            api_key=api_key
        )
