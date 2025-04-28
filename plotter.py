import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plotter.py file.csv")
        return

    csv_file = sys.argv[1]

    try:
        data = pd.read_csv(csv_file, header=None)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return

    if data.shape[1] < 2:
        print(f"Error: Expected at least 2 columns in {csv_file}")
        return

    x = data[0]
    y = data[1]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Plot of {csv_file}')
    plt.grid(True)

    output_file = csv_file.replace('.csv', '.png')
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

    # Try to open the image file
    if sys.platform.startswith('linux'):
        os.system(f'xdg-open {output_file}')
    elif sys.platform.startswith('darwin'):
        os.system(f'open {output_file}')
    elif sys.platform.startswith('win'):
        os.system(f'start {output_file}')

if __name__ == "__main__":
    main()
