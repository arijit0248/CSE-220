## --uses Linear Convolution--##

import numpy as np
from discrete_framework import DiscreteSignal, FastFourierTransform


def nextPowerOfTwo(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def normalize_base10(coeffs):
    coeffs = coeffs.astype(np.int64, copy=True)
    carry = 0
    for i in range(len(coeffs)):
        total = coeffs[i] + carry
        coeffs[i] = total % 10
        carry = total // 10

    while carry != 0:
        coeffs = np.append(coeffs, carry % 10)
        carry //= 10

    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
    return coeffs


def multi(x, y):
    x_digits = [int(d) for d in str(x)][::-1]
    y_digits = [int(d) for d in str(y)][::-1]

    a = np.array(x_digits, dtype=np.complex128)
    b = np.array(y_digits, dtype=np.complex128)

    L = len(a) + len(b) - 1
    N = nextPowerOfTwo(L)

    signalA = DiscreteSignal(a).pad(N)
    signalB = DiscreteSignal(b).pad(N)

    fft = FastFourierTransform()
    A = fft.compute_dft(signalA)
    B = fft.compute_dft(signalB)

    c = fft.compute_idft(A * B)

    vals = np.real(c)
    vals[np.abs(vals) < 1e-6] = 0.0

    coeffs = np.rint(vals).astype(np.int64)[:L]
    coeffs = normalize_base10(coeffs)

    return "".join(str(d) for d in coeffs[::-1])


def main():
    x = "65767879797907"
    y = "765454532435435345"
    print(multi(x, y))


if __name__ == "__main__":
    main()
