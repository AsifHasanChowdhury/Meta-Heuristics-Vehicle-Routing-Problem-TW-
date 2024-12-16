import os
import io
from json import load

def filePath():
    return 'D:\GA\py-ga-VRPTW\data\json\C107.json'



def guess_path_type(path):
    if os.path.isfile(path):
        return 'File'
    if os.path.isdir(path):
        return 'Directory'
    if os.path.islink(path):
        return 'Symbolic Link'
    if os.path.ismount(path):
        return 'Mount Point'
    return 'Path'


#file path checking
def exist(path, overwrite=False, display_info=True):
    if os.path.exists(path):
        if overwrite:
            # if display_info:
            #     print(f'{guess_path_type(path)}: {path} exists. Overwrite.')
            os.remove(path)
            return False
        # if display_info:
            # print(f'{guess_path_type(path)}: {path} exists.')
        return True
    # if display_info:
    #     print(f'{guess_path_type(path)}: {path} does not exist.')
    return False


#converting json data to useful format
def load_instance(json_file):
    #Converted filePath generic for All
    if exist(path=filePath(), overwrite=False, display_info=True):
        # print("file exist")
        #Converted filePath generic for All
        with io.open(filePath(), 'r', encoding='utf-8', newline='') as file_object:
            return load(file_object)
    else:
        print("Check Your File Path")
    return None