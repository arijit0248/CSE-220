import numpy as np


class DiscreteSignal:
    """
    Represents a discrete-time signal.
    """

    def __init__(self, data):
        # Ensure data is a numpy array, potentially complex
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)

    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal object.
        """
        currentLength = len(self.data)
        if (new_length >= currentLength):
            padded = np.zeros(new_length, dtype=np.complex128)
            padded[:currentLength] = self.data
        else:
            padded = self.data[:new_length]
        return DiscreteSignal(padded)

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Required for Task 4 (Drawing App).
        """
        currentLength = len(self.data)
        if currentLength == 0:
            return DiscreteSignal(np.array([], dtype=np.complex128))
        if currentLength == 1:
            return DiscreteSignal(np.full(new_length, self.data[0], dtype=np.complex128))

        extended = np.concatenate([self.data, self.data[:1]])
        oldIndices = np.arange(currentLength+1, dtype=np.float64)
        newIndices = np.linspace(0, currentLength, new_length, endpoint=False)
        realInterpolated = np.interp(newIndices, oldIndices, extended.real)
        imagInterpolated = np.interp(newIndices, oldIndices, extended.imag)
        return DiscreteSignal(realInterpolated+1j*imagInterpolated)


class DFTAnalyzer:
    """
    Performs Discrete Fourier Transform using O(N^2) method.
    """

    def compute_dft(self, signal: DiscreteSignal):
        """
        Compute DFT using naive summation.
        Returns: numpy array of complex frequency coefficients.
        """
        x = signal.data
        N = len(x)

        X = np.zeros(N, dtype=np.complex128)
        for k in range(N):
            for n in range(N):
                X[k] += x[n] * np.exp(-1j*2*np.pi*k*n/N)
        return X

    def compute_idft(self, spectrum):
        """
        Compute Inverse DFT using naive summation.
        Returns: numpy array (time-domain samples).
        """
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        x = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            for k in range(N):
                x[n] += X[k] * np.exp(1j*2*np.pi*k*n/N)
            x[n] /= N
        return x


class FastFourierTransform(DFTAnalyzer):
    def fftRecursive(self, x):
        N = len(x)
        if N == 1:
            return x.copy()

        even = self.fftRecursive(x[0::2])
        odd = self.fftRecursive(x[1::2])
        k = np.arange(N//2)
        factor = np.exp(-1j*2*np.pi*k/N)
        val = factor * odd

        X = np.concatenate([even+val, even-val])
        return X

    def compute_dft(self, signal: DiscreteSignal):
        x = signal.data.astype(np.complex128)
        N = len(x)
        if N == 0:
            return np.array([], dtype=np.complex128)
        if N & (N-1) != 0:
            raise ValueError("FFT requires power of 2 length")
        return self.fftRecursive(x)

    def compute_idft(self, spectrum):
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        if N == 0:
            return np.array([], dtype=np.complex128)
        signal = DiscreteSignal(np.conj(X))
        result = self.fftRecursive(signal.data)
        return np.conj(result) / N


class BluesteinFFT(FastFourierTransform):
    def bluestein(self, x, sign):
        N = len(x)
        if N == 1:
            return x.copy()

        n = np.arange(N, dtype=np.float64)
        chirp = np.exp(1j*sign*np.pi*n*n/N)
        a = x*chirp

        M = 1
        while M < (2*N-1):
            M <<= 1

        b = np.zeros(M, dtype=np.complex128)
        b[:N] = np.conj(chirp)
        b[M-N+1:] = np.conj(chirp[1:])[::-1]

        A = self.fftRecursive(np.pad(a, (0, M-N)))
        B = self.fftRecursive(b)

        conv = np.conj(self.fftRecursive(np.conj(A*B)))/M
        return chirp*conv[:N]

    def compute_dft(self, signal):
        x = signal.data.astype(np.complex128)
        N = len(x)
        if N == 0:
            return np.array([], dtype=np.complex128)
        if N & (N-1) == 0:
            return self.fftRecursive(x)
        return self.bluestein(x, sign=-1)

    def compute_idft(self, spectrum):
        X = np.array(spectrum, dtype=np.complex128)
        N = len(X)
        if N == 0:
            return np.array([], dtype=np.complex128)
        if N & (N-1) == 0:
            signal = DiscreteSignal(np.conj(X))
            result = self.fftRecursive(signal.data)
            return np.conj(result) / N
        return self.bluestein(X, sign=1)/N
