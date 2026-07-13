"""Plot backends. Only matplotlib_backend.py may import matplotlib."""


def get_backend(name='matplotlib'):
    if name == 'matplotlib':
        from plotdata.fault_transect.backends.matplotlib_backend import MatplotlibBackend
        return MatplotlibBackend()
    raise ValueError(f'Unknown plot backend: {name}')
