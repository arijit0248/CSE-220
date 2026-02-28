import numpy as np
import matplotlib.pyplot as plt


class Signal:
    def __init__(self, INF):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1, dtype=float)

    # def __repr__(self):
    #     return f"{self.values}"

    def timeToIndex(self, t):
        return t + self.INF

    def set_value_at_time(self, t, value):
        if t < -self.INF or t > self.INF:
            return
        self.values[self.timeToIndex(t)] = float(value)

    def get_value_at_time(self, t):
        if t < -self.INF or t > self.INF:
            return
        return self.values[self.timeToIndex(t)]

    def shift(self, k):
        newINF = self.INF + abs(k)
        result = Signal(newINF)
        start = k + newINF - self.INF
        end = start + (2 * self.INF) + 1
        result.values[start:end] = self.values
        return result

    def add(self, other):
        maxINF = max(self.INF, other.INF)
        result = Signal(maxINF)

        selfStart = maxINF - self.INF
        selfEnd = selfStart + len(self.values)

        otherStart = maxINF - other.INF
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
        # plt.savefig("Input Signal")
        plt.show()


class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.h = impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        impulses = []
        coefficients = []
        for k in range(-input_signal.INF, input_signal.INF + 1):
            val = input_signal.get_value_at_time(k)
            if val != 0:
                impulse = Signal(input_signal.INF)
                impulse.set_value_at_time(k, 1)
                impulses.append(impulse)
                coefficients.append(val)
        return impulses, coefficients

    def output(self, input_signal: Signal):
        outputSignal = Signal(input_signal.INF)
        impulses, coefficients = self.linear_combination_of_impulses(
            input_signal)
        for i in range(len(impulses)):
            impulse = impulses[i]
            coefficient = coefficients[i]
            k = 0
            for t in range(-impulse.INF, impulse.INF + 1):
                if impulse.get_value_at_time(t) != 0:
                    k = t
                    break
            hShifted = self.h.shift(k)
            hScaled = hShifted.multiply(coefficient)
            outputSignal = outputSignal.add(hScaled)
        return outputSignal


if __name__ == "__main__":
    INF = 10

    # Input signal x(n)
    x = Signal(INF)
    x.set_value_at_time(-2, 1)
    x.set_value_at_time(0, 2)
    x.set_value_at_time(3, -1)

    x.plot("Input Signal x(n)")

    # Impulse response h(n)
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    h.plot("Impulse Response h(n)")

    # LTI System
    system = LTI_System(h)

    # Output
    y = system.output(x)
    y.plot("Output Signal y(n)")
