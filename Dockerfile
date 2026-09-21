FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8080

WORKDIR /app

COPY app/ /app/app/

# Pure-stdlib service: no third-party packages to install.
EXPOSE 8080

# Container-level health check queries the service over HTTP.
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-m", "app.healthcheck"]

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

CMD ["python", "-m", "app.server"]
