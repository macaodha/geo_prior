def get_paths(variable_name):
    paths = {
    'mask_dir': '../data/',
    'inat_2017_data_dir': '../data/inat_2017/',
    'inat_2018_data_dir': '../data/inat_2018/',
    'birdsnap_data_dir': '../data/birdsnap/',
    'nabirds_data_dir': '../data/nabirds/',
    'yfcc_data_dir': '../data/yfcc/'
    }
    return paths[variable_name]
