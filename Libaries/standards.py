import os


# Function to create a results folder with a specified path and name
def results_folder(path, name, db=None):
    # Determine the save directory and folder name based on the presence of a database name
    if db is not None:
        save_dir = f'{path}/{name}_{db}'
        temp = f'{name}_{db}'
    else:
        save_dir = f'{path}/{name}'
        temp = f'{name}'

    try:
        # Check if the directory already exists
        if os.path.exists(save_dir):
            pass
        else:
            # Create the directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
    
    except (OSError, FileExistsError) as e:
        # Handle potential UnboundLocalError
        print('Error occurred')
    return save_dir

def join_path(path1, path2):
    return os.path.join(path1, path2)


def paths(path):
    # Path to where the code is stored
    path_github = join_path(path, r'RA\Single-use-vs-multi-use-in-health-care')
    # Specifying the LCIA method

    ecoinevnt_paths = {'ev391apos' : join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_apos_ecoSpold02\datasets"),
                    'ev391consq' :   join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_consequential_ecoSpold02\datasets"),
                    'ev391cutoff' :  join_path(path, r"4. semester\EcoInvent\ecoinvent 3.9.1_cutoff_ecoSpold02\datasets")}
    system_path = [
            join_path(path_github, r'Data\databases\case1.xlsx'), 
            join_path(path_github, r"Data\databases\case2.xlsx")
                ]
    
    return path_github, ecoinevnt_paths, system_path
