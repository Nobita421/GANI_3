import logging
import os
from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_dir: str):
    """Setup logger and tensorboard writer."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Python logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    return logger, writer
