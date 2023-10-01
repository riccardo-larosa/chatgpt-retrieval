
# Python script to loop through the color of the terminal
import os

for i in range(1, 100):
    print(f"\033[38;5;{i}m this is {i}\033[0m")