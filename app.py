import logging
import scrygent

# Configure the global logging format at the entrypoint
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Initializing scrygent pipeline entry point...")
    message = scrygent.hello()
    print(f"Output: {message}")
    logger.info("Pipeline test execution complete.")

if __name__ == "__main__":
    run_pipeline()