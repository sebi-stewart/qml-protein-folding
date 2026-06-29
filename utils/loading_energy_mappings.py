"""
Load saved energy-mapping JSON files and convert them into Python tensor dictionaries.
Parses one-body and two-body sections into integer-indexed structures for downstream use.
"""

import json

def load_energy_mappings(source_path):
    with open(source_path, 'r') as f:
        raw = json.load(f)
    one_body = {int(k): {int(rk): rv for rk, rv in v.items()} for k, v in raw['one_body'].items()}
    two_body = {
        tuple(int(x) for x in k.split(',')): {
            tuple(int(x) for x in rk.split(',')): rv
            for rk, rv in interactions.items()
        }
        for k, interactions in raw['two_body'].items()
    }
    return one_body, two_body