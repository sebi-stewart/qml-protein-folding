"""
Initialize PyRosetta with conservative flags to avoid unrecognized residues and other junk.
This function centralises the initialisation flags used across extraction workflows.
"""

def initialize_rosetta(pyrosetta, extra_flags: str) -> None:

    # -ignore_unrecognized_res --> Skips drugs, weird metals, or unknown amino acids
    clean_flags = "-ignore_unrecognized_res"
    all_flags = f"{clean_flags} {extra_flags}"

    print(f"Initializing PyRosetta with cleaning flags: {clean_flags}" +
          f" and extra flags: {extra_flags}" if extra_flags else "")

    pyrosetta.init(all_flags)