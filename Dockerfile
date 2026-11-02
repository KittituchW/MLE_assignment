FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts
ENV RECO_ARTIFACT=/app/artifacts/model
USER 65532
EXPOSE 8000
CMD ["uvicorn", "discovery_reco.api:app", "--host", "0.0.0.0", "--port", "8000"]
