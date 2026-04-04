def initialize_rosetta(pyrosetta, extra_flags: str) -> None:
    """
    Initializes PyRosetta with strict flags to ignore junk data,
    then loads and returns a clean Pose object.
    """

    # -ignore_unrecognized_res --> Skips drugs, weird metals, or unknown amino acids
    clean_flags = "-ignore_unrecognized_res"
    all_flags = f"{clean_flags} {extra_flags}"

    print(f"Initializing PyRosetta with cleaning flags: {clean_flags}" +
          f" and extra flags: {extra_flags}" if extra_flags else "")

    pyrosetta.init(all_flags)