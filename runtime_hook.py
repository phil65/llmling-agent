import os


# Disable Pydantic plugins (logfire) to avoid source code inspection issues in frozen apps
os.environ["PYDANTIC_DISABLE_PLUGINS"] = "1"
