import importlib

def reload_lib(lib):
    # https://dev.to/fronkan/importlib-reload-for-resting-modules-between-tests-neh
    for l in lib:
        importlib.reload(l)
    