from __future__ import print_function
import os
import hydra
import logging
from agents import *
from logger import *
# from utils.config import *

@hydra.main(config_path='../conf/config.yaml')
def main(config):
    # Initialize Logger
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    log = logging.getLogger('[MAIN]')  
    log.info('Initialize Butane Runtime Engine')

    # Display Configuration
    log.info('[Runtime Configuration]')
    log.info(config.pretty())

    # Initialize Logger
    log.info('Initialize Logger Object: ' + config.logger.name)
    logger = eval(config.logger.name).Logger(config)

    # Generate and Initialize Agent
    agent = eval(config.agent.name).Agent(config, logger)
    agent.run()
    #agent.finalize()

    # Note: If optimization routine is used, we must return something here...
    # return config.batch_size

if __name__ == '__main__':
    main()