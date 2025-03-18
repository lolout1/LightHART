import sys
import traceback
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("debug_logs", "training_wrapper.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_wrapper")

def run_training():
    try:
        # Get arguments from command line
        args = sys.argv[1:]
        
        # Start timer
        start_time = time.time()
        logger.info(f"Starting training with args: {' '.join(args)}")
        
        # Import trainer from main.py and run
        sys.path.append(os.getcwd())
        from main import main
        main(args)
        
        # Log completion time
        elapsed = time.time() - start_time
        logger.info(f"Training completed successfully in {elapsed/60:.2f} minutes")
        return 0
        
    except Exception as e:
        logger.error("Training failed with exception:")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = run_training()
    sys.exit(exit_code)
