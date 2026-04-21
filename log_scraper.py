import pathlib
from collections import defaultdict
import pandas as pd

log_files = pathlib.Path("QAOA_Results_Backup_3").rglob("*.log")

runtimes = defaultdict(list)

for log_file in log_files:
    with open(log_file, "r") as f:

        cur_qubit_count: int | None = None
        cur_qaoa_layers: set[int] = set()

        while line := f.readline():
            if not cur_qubit_count and "Reduced Hamiltonian built:" in line:
                stripped_line = line.strip()
                parts = stripped_line.split(" ")
                qubits_part = parts[-5] if len(parts) >= 5 else None
                if qubits_part:
                    cur_qubit_count = int(qubits_part)

            elif cur_qubit_count:
                if "qaoa.main.p_" in line:
                    logger_name = line.strip().split(" ")[2] if len(line.strip().split(" ")) >= 3 else None
                    if logger_name and logger_name.startswith("qaoa.main.p_"):
                        qaoa_layers = int(logger_name.split("_")[-1])
                        cur_qaoa_layers.add(qaoa_layers)

                elif "Completed QAOA runs for" in line:
                    if not cur_qaoa_layers:
                        print(f"Warning: No QAOA layer info found for {log_file.name} with {cur_qubit_count} qubits.")

                    layer_count = sum(list(cur_qaoa_layers))

                    stripped_line = line.strip().split("-")[1].strip()
                    parts = stripped_line.split(" ")
                    time_part = parts[-2] if len(parts) >= 2 else None
                    if time_part:
                        runtimes[cur_qubit_count].append((float(time_part), layer_count))

                    cur_qubit_count = None
                    cur_qaoa_layers = set()


print("Qubits | Avg Time (s) | Runs")
for qubits, tup in sorted(runtimes.items()):
    times, layer_counts = zip(*tup)

    avg_time = sum(times) / len(times)
    print(f"{qubits:6} | {avg_time:12.2f} | {len(times)}")

print("\nRuntimes per layer grouped by qubit count:")
print("Qubits | Total Layers | Avg Time (s) | Runs")
for qubits, tup in sorted(runtimes.items()):
    times, layer_counts = zip(*tup)

    total_layers = sum(layer_counts)
    avg_time = sum(times) / len(times)
    avg_time_per_layer = avg_time / total_layers if total_layers > 0 else float('inf')

    print(f"{qubits:6} | {total_layers:12} | {avg_time_per_layer:12.4f} | {len(times)}")

print("Populating DataFrame for detailed analysis...")
data = []
for qubits, tup in sorted(runtimes.items()):
    for time, layer_count in tup:
        data.append({"Qubits": qubits, "Total Layers": layer_count, "Time (s)": time})
df = pd.DataFrame(data)
print(df)
df.to_pickle("colab_runtimes2.pkl")