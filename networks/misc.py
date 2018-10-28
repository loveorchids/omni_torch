
def get_stride_padding_kernel(input, out):
    """
    Calculation based on:
    https://www.quora.com/How-can-I-calculate-the-size-of-output-of-convolutional-layer
    :param input:
    :param out:
    :return:
    """
    for K in range(1, 6):
        for S in range(1, 6):
            for P in range(6):
                if float(P) == (input - 1 - S * (out - 1)) / 2:
                    return S, P, K