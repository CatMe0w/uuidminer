import csv
import uuid
import sys
import os


def analyze_results(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # dictionary to store the best results for each length
    # key: length (int)
    # value: list of (uuid_int, uuid_str, username)
    best_results = {}

    print(f"Reading {filename}...")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue

                full_name = row[0]
                uuid_str = row[1]

                length = len(full_name)

                try:
                    # parse UUID and get integer value (as i128) for comparison
                    uuid_val = uuid.UUID(uuid_str)
                    uuid_int = uuid_val.int

                    if length not in best_results:
                        best_results[length] = []
                    
                    best_results[length].append((uuid_int, uuid_str, full_name))

                except ValueError:
                    print(f"Skipping invalid UUID: {uuid_str}")
                    continue

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # sort and keep top 10 for each length
    for length in best_results:
        best_results[length].sort(key=lambda x: x[0])
        best_results[length] = best_results[length][:10]

    print("\n" + "=" * 64)
    print(f"{'Length':<6} | {'Username':<16} | {'UUID'}")
    print("=" * 64)

    sorted_lengths = sorted(best_results.keys())

    for length in sorted_lengths:
        print(f"--- Length {length} ---")
        for _, uuid_str, full_name in best_results[length]:
            print(f"{length:<6} | {full_name:<16} | {uuid_str}")

    print("=" * 64)


if __name__ == "__main__":
    filename = "results.csv"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    analyze_results(filename)
