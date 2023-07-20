import yaml


class YamlReader:
    """
    Class for parsing YAML files.
    """

    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        self.config = self.load_yaml()

    def load_yaml(self):
        with open(self.yaml_file, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError:
                raise IOError("Invalid .yaml/.yml file.")
        return config
