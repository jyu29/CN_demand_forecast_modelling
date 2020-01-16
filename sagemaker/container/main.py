config_file = vars(args)['config_file']
    log_config_file = vars(args)['log_file']
    # Optional arguments
    dir_name = vars(args)['dir_path']
    max_words = vars(args)['max_words']
    
    # Configure the whole program (logging, external config files, singletons, ...)
    config.configure_logging(log_config_file)
    config.pgconf = ProgramConfiguration(config_file)