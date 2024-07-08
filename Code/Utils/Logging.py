import logging


def setup_logging(file_name):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(file_name), logging.StreamHandler()],
                        force=True)