import pathlib
from collections import defaultdict
import pandas as pd

log_files = list(pathlib.Path("temp4").rglob("*.log*"))

runtimes = defaultdict(list)

file_name_map = {
    "No_adjoint_5_8_qubits": "No Adjoint",
    "No_jit_5_8_qubits": "No JIT",
    "No_batching_5_8_qubits": "No Batching"
}

print("Processing log files...", log_files)

for log_file in log_files:
    for key, readable_name in file_name_map.items():
        if key in log_file.name:
            file_type = readable_name
            break
    else:
        file_type = "Unknown"

    with open(log_file, "r") as f:

        cur_qubit_count: int | None = None

        while line := f.readline():
            if not cur_qubit_count and "Reduced Hamiltonian built:" in line:
                stripped_line = line.strip()
                parts = stripped_line.split(" ")
                qubits_part = parts[-5] if len(parts) >= 5 else None
                if qubits_part:
                    cur_qubit_count = int(qubits_part)

            elif cur_qubit_count:
                if "Completed QAOA runs for" in line:

                    stripped_line = line.strip().split("-")[1].strip()
                    parts = stripped_line.split(" ")
                    time_part = parts[-2] if len(parts) >= 2 else None
                    if time_part:
                        runtimes[cur_qubit_count].append((float(time_part), file_type))

                    cur_qubit_count = None


print("Qubits | Avg Time (s) | Runs")
for qubits, tup in sorted(runtimes.items()):
    times, layer_counts = zip(*tup)

    avg_time = sum(times) / len(times)
    print(f"{qubits:6} | {avg_time:12.2f} | {len(times)}")

print("Populating DataFrame for detailed analysis...")
data = []
for qubits, tup in sorted(runtimes.items()):
    for time, file_type in tup:
        data.append({"Qubits": qubits, "FileType": file_type, "Time (s)": time})
df = pd.DataFrame(data)
print(df)
df.to_pickle("qaoa_30seed_constansts.pkl")