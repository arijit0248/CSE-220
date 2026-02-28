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
        self.values[self.timeToIndex(t)] = float(value)

    def get_value_at_time(self, t):
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


# Stock Market Prices as a Python List
price_list = list(map(int, input("Stock Prices: ").split()))
n = int(input("Window size: "))
alpha = float(input("Alpha: "))

# # You may use the following input for testing purpose
# price_list = [10, 11, 12, 9, 10, 13, 15, 16, 17, 18]
# n = 3
# alpha = 0.8

INF = len(price_list)
inputSignal = Signal(INF)
for i in range(INF):
    inputSignal.set_value_at_time(i, price_list[i])

impulseSignal = Signal(n)
for i in range(n):
    coeff = alpha * pow((1 - alpha), i)
    impulseSignal.set_value_at_time(i, coeff)

lti = LTI_System(impulseSignal)
outputSignal = lti.output(inputSignal)
# Determine the values after performing Exponential Smoothing
# The length of exsm should be = len(price_list) - n + 1
exsm = []
for i in range(n-1, len(price_list)):
    exsm.append(outputSignal.get_value_at_time(i))

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63
