import pyrosetta
from benchmark import bpti_ryfyn_benchmark
from rotamer_extraction import extract_top_n_rotamers
from hamiltonian_creation import extract_hamiltonian_tensors, build_ising_hamiltonian
from initialisation import initialize_rosetta


if __name__ == '__main__':
    initialize_rosetta(pyrosetta, extra_flags="-mute all")

    benchmark_pose = bpti_ryfyn_benchmark()
    rotamer_lib, ig, rot_sets = extract_top_n_rotamers(benchmark_pose)
    # h_linear, J_quadratic = extract_hamiltonian_tensors(rotamer_lib, ig, rot_sets)
    # H_target = build_ising_hamiltonian(h_linear, J_quadratic, penalty=500.0)
    # print(H_target)
