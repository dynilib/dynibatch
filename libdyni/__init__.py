import logging
import json

# basic logging config
try:
    with open('logging.json', 'rb') as f:
        logging.config.dictConfig(json.load(f))
except:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M")
