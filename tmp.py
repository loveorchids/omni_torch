import math
import numpy as np

def measure(num, ratio):
    v_num = int(max(math.sqrt(num) * ratio, 1))
    h_num = math.ceil(num / v_num)
    if v_num * h_num < num:
        print(v_num * h_num)
        print("error at num: %d, ratio: %f" %(num, ratio))


if __name__ == "__main__":
    a = np.zeros((3, 50, 50))
    b = np.zeros((50, 50, 1))
    b[:,:,:] = a[1, :, :]
