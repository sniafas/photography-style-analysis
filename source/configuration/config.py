import os
import sys
import json


class Configuration:
    """
    Configuration reads and return the configuration.json file with the essential settings
    """

    def __init__(self):

        self.config = self.create_configuration()

    def get_configuration(self):
        return self.config

    def create_configuration(self):
        """
        Reads .json file and returns a dictionary with configuration
        """
        configuration_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'configuration'
        )
        config_file = open(os.path.join(
            configuration_dir, 'configuration.json'), 'r')
        config = json.load(config_file)

        return config
