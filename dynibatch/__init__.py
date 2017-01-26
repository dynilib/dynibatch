import os
import logging.config
import json

# basic logging config
LOGGING_FILENAME = os.path.join(os.path.dirname(__file__), '../logging.json')
try:
    with open(LOGGING_FILENAME, 'r') as f:
        logging.config.dictConfig(json.load(f))
except IOError:
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M")
    logging.debug('cannot open %s', LOGGING_FILENAME)
