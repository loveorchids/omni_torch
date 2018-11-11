import math
import numpy as np

def measure(num, ratio):
    v_num = int(max(math.sqrt(num / ratio), 1))
    h_num = math.ceil(num / v_num)
    if v_num * h_num < num:
        print(v_num * h_num)
        print("error at num: %d, ratio: %f" %(num, ratio))


if __name__ == "__main__":
    for num in range(4, 1000):
        for r in range(2, 50):
            measure(num, r/10)
