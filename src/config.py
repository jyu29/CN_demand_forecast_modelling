from datetime import datetime
import yaml


class ProgramConfiguration(object):
    """
    Class used to handle and maintain all parameters of this program (timeouts, some other values...)
    """
    _config = None
    _config_directory = None

    def __init__(self, config_file_path):
        """
        Constructor - Loads the given external YAML configuration file. Raises an error if not able to do it.
        :param config_file_path: (string) full path to the YAML configuration file
        """
        if os.path.exists(config_file_path):
            self._config_directory = os.path.dirname(config_file_path)
            with open(config_file_path, 'r') as f:
                self._config = yaml.load(f)
        else:
            raise Exception("Could not load external YAML configuration file '{}'".format(config_file_path))
            
    def get_global_tags(self):
        return self._config['global']['tags']
    
    def get_global_role_arn(self):
        return self._config['global']['role_arn']
    
    def get_global_security_group_ids(self):
        return self._config['global']['security_group_ids'].split(', ')
    
    def get_global_subnets(self):
        return self._config['global']['subnets'].split(', ')
    
    def get_train_bucket_input(self):
        return self._config['train']['bucket_input']
    
    def get_train_bucket_output(self):
        return self._config['train']['bucket_output']
    
    def get_train_path_refined_data_input(self):
        return self._config['train']['path_refined_data_input']
    
    def get_train_path_refined_data_output(self):
        return self._config['train']['path_refined_data_output']
    
    def get_train_image(self):
        return self._config['train']['docker_image']
    
    def get_train_job_name(self):
        now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        return self._config['train']['job_name'] + '-' + now_str
    
    def get_train_instance_type(self):
        return self._config['train']['instance_type']
    
    def get_train_instance_count(self):
        return self._config['train']['instance_count']