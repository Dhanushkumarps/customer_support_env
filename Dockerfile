# =============================================================================
# Dockerfile — Customer Support OpenEnv Environment
# =============================================================================
# Build and run locally:
#   docker build -t customer_support_env .
#   docker run -p 7860:7860 customer_support_env
#
# Deploy to Hugging Face Spaces:
#   docker build && docker push to your HF Space repo
# =============================================================================

FROM python:3.10-slim

# ---------- Labels ---------- #
LABEL maintainer="your-hf-username"
LABEL version="0.1.0"
LABEL description="OpenEnv customer support simulation environment for AI agent evaluation."

# ---------- Working directory ---------- #
WORKDIR /app

# ---------- Install dependencies first (layer caching) ---------- #
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy project files ---------- #
COPY . .

# ---------- Expose HF Spaces port ---------- #
EXPOSE 7860

# ---------- Health check ---------- #
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# ---------- Start the server ---------- #
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
