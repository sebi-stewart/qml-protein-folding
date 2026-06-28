from pathlib import Path

outputs_folder = Path("./outputs/")

def navigate_and_delete_checkpoints(base_folder: Path):
    # Search all direct items in the base folder
    for item in base_folder.iterdir():
        if item.is_dir():
            # If it's a directory, recursively call the function
            navigate_and_delete_checkpoints(item)
        elif item.is_file() and item.name.endswith("checkpoint.pkl"):
            file_name = item.name
            prefix = file_name.split("_layers_")[0]
            final_file = f"{prefix}_final.pkl"
            if item.parent.joinpath(final_file).exists():
                print("Removing checkpoint file as final file exists:", item)
                item.unlink()  # Delete the checkpoint file
            else:
                print("Final file does not exist for checkpoint:", item)
                # item.unlink()  # Delete the checkpoint file
            # If it's a file and starts with the prefix, delete it
            # print(f"Deleting checkpoint file: {item}")
            # item.unlink()  # Delete the file
# navigate_and_delete_checkpoints(outputs_folder)

def find_duplicate_tests(base_folder: Path):
    # A test starts with the compresssed name of the test, which is everything before the AF-5PTI part of the name.
    # A test result ends with _layers.npz
    test_instance_map = {}
    test_batch_map = {}
    tests = list(base_folder.rglob("*_layers.npz"))
    tests = [test for test in tests if not test.name.startswith("Oldruns")]

    for test in tests:
        compressed_instance, test_name = test.name.split("_AF-5PTI_")

        # the compressed instance can be shortened more, it consists of _x_metrics, where x is the instance number it was run on. We can remove the _x_metrics part to get the original test name.
        # We do not need the instance name so we remove it. The instance name is everything after the last underscore in the compressed instance name.
        test_batch = "_".join(compressed_instance.split("_")[:-2])
        test_computer_instance = compressed_instance.split("_")[-2]
        filtered_test_name = "_".join(test_name.split("_")[:-2])

        print(f"Batch: {test_batch}, Computer Instance: {test_computer_instance}, Filtered Test Name: {filtered_test_name}")

        if filtered_test_name not in test_instance_map:
            test_instance_map[filtered_test_name] = set()
        test_instance_map[filtered_test_name].add(test_computer_instance)

        if filtered_test_name not in test_batch_map:
            test_batch_map[filtered_test_name] = set()
        test_batch_map[filtered_test_name].add(test_batch)

    # If there are no duplicate tests, then the set of computer instances for each test name should have a length of 1. If any test name has a set of computer instances with a length greater than 1, then that test has been run on multiple computer instances and is a duplicate.
    any_broke = False
    for test_name, computer_instances in test_instance_map.items():
        if len(computer_instances) > 1:
            print(f"Duplicate test found: {test_name} has been run on multiple computer instances: {computer_instances}")
            any_broke = True

    for test_name, test_batches in test_batch_map.items():
        if len(test_batches) > 1:
            print(f"Duplicate test found: {test_name} has been run on multiple batches: {test_batches}")
            any_broke = True

    if not any_broke:
        print("\nNo duplicate tests found.")

        # Since tests are unique and there are no duplicates, we can compress the file paths to make them easier to navigate.
        for test in tests:
            _, test_name = test.name.split("_AF-5PTI_")
            new_test_name = f"AF-5PTI_{test_name}"
            test_path = test.parent.joinpath(new_test_name)
            test.rename(new_test_name)


def compress_file_paths(base_folder: Path):
    """Flatten all files under *base_folder* into its parent directory.

    Example:
        base/folder1/folder2/file.txt -> base_folder_parent/base_folder_folder1_folder2_file.txt

    The special directory name ``phase2_5_to_10_qubits`` is compressed as
    ``metrics`` instead.
    """
    base_folder = base_folder.resolve()
    destination_root = base_folder.parent

    # Move files first so directory deletion can happen bottom-up afterwards.
    for item in sorted((p for p in base_folder.rglob("*") if p.is_file())):
        relative_parts = ["metrics" if part == "phase2_5_to_10_qubits" else part for part in item.relative_to(destination_root).parts]
        new_file_name = "_".join(relative_parts)
        new_file_path = destination_root.joinpath(new_file_name)
        print(f"Renaming {item} to {new_file_path}")
        item.rename(new_file_path)

    # Remove empty directories from the deepest path back to the base folder.
    directories = sorted((p for p in base_folder.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for directory in directories:
        print(f"Removing directory: {directory}")
        directory.rmdir()

    print(f"Removing base directory: {base_folder}")
    base_folder.rmdir()


folders_to_compress = [
    Path("outputs/comparisons_of_runtimes/temp"),
    Path("outputs/comparisons_of_runtimes/temp2"),
    Path("outputs/comparisons_of_runtimes/temp3"),
    Path("outputs/comparisons_of_runtimes/temp4"),
]

# for folder in folders_to_compress:
#     compress_file_paths(folder)

# find_duplicate_tests(Path("./outputs/phase2"))

import time
def get_last_modified_of_folder(folder: Path):
    """Get the last modified time of a folder and its contents."""
    files = list(folder.rglob("*"))
    most_recent_time = 0
    for file in files:
        if not file.is_file(): continue
        if file.name.startswith(".DS_Store"): continue
        last_modified_time = file.stat().st_mtime
        if last_modified_time > most_recent_time:
            most_recent_time = last_modified_time
            # print(f"New most recent file: {file}, last modified time: {time.ctime(last_modified_time)}")
    print(f"Last modified time of folder {folder}: {time.ctime(most_recent_time)}")


result_folders = [folder for folder in Path("./outputs/").iterdir() if folder.is_dir()]
for folder in result_folders:
    get_last_modified_of_folder(folder)