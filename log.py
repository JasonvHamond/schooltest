import logging

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)

# Create a handler for writing to a file
file_handler = logging.FileHandler('log.txt')

# Create a handler for writing to the console
console_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log messages
logger.info('Starting program...')
logger.info('Loading data...')
logger.info('Training model...')
logger.info('Making predictions...')
logger.warning('Low confidence in assigned class label!')
logger.info('Finished!')
