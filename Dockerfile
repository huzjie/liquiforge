# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY liquiforge ./liquiforge
COPY scripts ./scripts

RUN pip install --no-cache-dir .

EXPOSE 8787

ENTRYPOINT ["liquiforge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8787"]
