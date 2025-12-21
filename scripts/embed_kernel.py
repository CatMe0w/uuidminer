import sys
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: embed_kernel.py <input_file> <output_header_file> <variable_name>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    variable_name = sys.argv[3]

    try:
        with open(input_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        sys.exit(1)

    header_content = f"""#pragma once

static const char* {variable_name} = R"EMBEDDED_SOURCE(
{content}
)EMBEDDED_SOURCE";
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header_content)

if __name__ == "__main__":
    main()
