from datetime import datetime
import yaml
import os

class ProgramConfiguration(object):
    """
    Class used to handle and maintain all configurations of this program
    """
    _config_tech = None
    _config_func = None
    
    def __init__(self, config_tech_path, config_func_path):
        """
        Constructor - Loads the given external YAML configuration file. Raises an error if not able to do it.
        :param config_tech_file_path: (string) full path to the YAML configuration file
        :param config_func_file_path: (string) full path to the YAML configuration file
        """
        if os.path.exists(config_tech_path):
            with open(config_tech_path, 'r') as f:
                self._config_tech = yaml.load(f)
        else:
            raise Exception("Could not load external YAML configuration file '{}'".format(config_tech_path))
           
        if os.path.exists(config_func_path):
            with open(config_func_path, 'r') as f:
                self._config_func = yaml.load(f)
        else:
            raise Exception("Could not load external YAML configuration file '{}'".format(config_func_path))
    
    
    def get_scope(self):
        return self._config_func['scope']
    
    def get_horizon(self):
        return self._config_func['horizon']
    
    def get_horizon_freq(self):
        return self._config_func['horizon_freq']
    
    def get_prediction_length(self):
        return self._config_func['prediction_length']
    
    def get_prediction_freq(self):
        return self._config_func['prediction_freq']
    
    def get_season_length(self):
        return self._config_func['season_length']
    
    ###
    
    
    def get_global_tags(self):
        return self._config_tech['global']['tags']
    
    def get_global_role_arn(self):
        return self._config_tech['global']['role_arn']
    
    def get_global_security_group_ids(self):
        return self._config_tech['global']['security_group_ids'].split(', ')
    
    def get_global_subnets(self):
        return self._config_tech['global']['subnets'].split(', ')
    
    def get_train_volume_size_in_gb(self):
        return self._config_tech['global']['volume_size_in_gb']
    
    def get_train_bucket_input(self):
        return self._config_tech['train']['bucket_input']
    
    def get_train_bucket_output(self):
        return self._config_tech['train']['bucket_output']
    
    def get_train_path_refined_data_input(self):
        return self._config_tech['train']['path_refined_data_input']
    
    def get_train_path_refined_data_intermediate(self):
        return self._config_tech['train']['path_refined_data_intermediate']
    
    def get_train_path_refined_data_output(self):
        return self._config_tech['train']['path_refined_data_output']

    def get_train_path_active_sales(self):
        return self._config_tech['train']['path_active_sales']
    
    def get_train_image_name(self):
        return self._config_tech['train']['image_name']
    
    def get_train_docker_image(self):
        return self._config_tech['train']['docker_image']
    
    def get_train_job_name(self):
        now_str = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
        return self._config_tech['train']['job_name'] + '-' + now_str
    
    def get_train_instance_type(self):
        return self._config_tech['train']['instance_type']
    
    def get_train_instance_count(self):
        return self._config_tech['train']['instance_count']

    def get_train_max_run_time(self):
        return self._config_tech['train']['max_run_time']
    
    def get_train_hyperparameters(self):
        hyperparameters = self._config_tech['train']['hyperparameters']
        for key in hyperparameters:
            hyperparameters[key] = str(hyperparameters[key])
        return hyperparameters

    def get_monitor_sleep(self):
        return self._config_tech['monitor']['sleep']