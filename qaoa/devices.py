import pennylane as qml

_DEVICE_CACHE = {}

def get_cached_device(num_qubits, device_type):
    cache_key = (num_qubits, device_type)
    if cache_key not in _DEVICE_CACHE:

        _DEVICE_CACHE.clear()
        _DEVICE_CACHE[cache_key] = qml.device(device_type, wires=range(num_qubits))
    return _DEVICE_CACHE[cache_key]
