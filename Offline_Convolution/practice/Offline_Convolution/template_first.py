import numpy as np
import matplotlib.pyplot as plt


class Signal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1, dtype=float)

    def time_to_index(self, t):
        return t + self.INF

    def set_value_at_time(self, t, value):
        if t < -self.INF or t > self.INF:
            return
        self.values[self.time_to_index(t)] = float(value)

    def shift(self, k):
        newINF = self.INF + abs(k)
        result = Signal(newINF)
        start = k + newINF - self.INF
        end = start + (2 * self.INF) + 1
        result.values[start:end] = self.values
        return result

    def add(self, other):
        newINF = max(self.INF, other.INF)
        result = Signal(newINF)

        selfStart = newINF - self.INF
        selfEnd = selfStart + len(self.values)

        otherStart = newINF - other.INF
        otherEnd = otherStart + len(other.values)

        result.values[selfStart:selfEnd] += self.values
        result.values[otherStart:otherEnd] += other.values

        return result

    def multiply(self, scalar):
        result = Signal(self.INF)
        result.values = self.values * scalar
        return result

    def plot(self, title="Discrete Signal"):
        time = np.arange(-self.INF, self.INF + 1)
        plt.figure(figsize=(12, 5))
        plt.stem(time, self.values)
        plt.xlabel("n")
        plt.ylabel("x[n]")
        plt.title(title)
        plt.grid(True)
        plt.show()


# class LTI_System:
#     def __init__(self, impulse_response: Signal):
#         # Initialize

#     def linear_combination_of_impulses(self, input_signal: Signal):
#         # Decompose the signal into impulses and corresponding coefficients

#     def output(self, input_signal: Signal):
#         # Calculate and return the output signal


if __name__ == "__main__":
    INF = 10

    # Input signal x(n)
    x = Signal(INF)
    x.set_value_at_time(-2, 1)
    x.set_value_at_time(0, 2)
    x.set_value_at_time(3, -1)

    x.plot("Input Signal x(n)")

    # # Impulse response h(n)
    # h = Signal(INF)
    # h.set_value_at_time(0, 1)
    # h.set_value_at_time(1, 0.5)

    # h.plot("Impulse Response h(n)")

    # # LTI System
    # system = LTI_System(h)

    # # Output
    # y = system.output(x)
    # y.plot("Output Signal y(n)")
