#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plotter.py file.csv")
        return

    csv_file = sys.argv[1]

    # load exactly as before
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

    # draw plot exactly as before
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'Plot of {csv_file}')
    plt.grid(True)

    # just display and block until you close the window
    plt.show()

if __name__ == "__main__":
    main()
