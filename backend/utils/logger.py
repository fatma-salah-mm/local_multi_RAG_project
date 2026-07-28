import logging
import sys

# Configure the basic logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create the specific logger instance that routes.py is looking for
app_logger = logging.getLogger("Enterprise_RAG")