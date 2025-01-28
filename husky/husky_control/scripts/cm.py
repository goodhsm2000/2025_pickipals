import subprocess

def load_new_map(map_yaml_path):

    subprocess.Popen(['rosnode', 'kill', '/map_server'])

    subprocess.Popen(['rosrun', 'map_server', 'map_server', map_yaml_path])

    # load_new_map('/path/to/new_map.yaml')
    
load_new_map("/home/turtle/woojungwon_01.yaml")