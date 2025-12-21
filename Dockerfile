FROM python:3.13-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir ".[default,server]"

# Expose the default port
EXPOSE 8000

# Run the server
ENTRYPOINT ["agentpool", "serve-api"]
CMD ["--auto-discover"]
