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


# Input for first polynomial
d1 = int(input("Degree of the first polynomial: "))
poly1 = list(map(int, input("Coefficients: ").split()))


# Input for second polynomial
d2 = int(input("Degree of the second polynomial: "))
poly2 = list(map(int, input("Coefficients: ").split()))

INF = d1 + d2 + 10
inputSignal = Signal(INF)
poly1.reverse()
for i in range(len(poly1)):
    inputSignal.set_value_at_time(i, poly1[i])
impulse = Signal(INF)
poly2.reverse()
for i in range(len(poly2)):
    impulse.set_value_at_time(i, poly2[i])

lti = LTI_System(impulse)
output = lti.output(inputSignal)

resultingCoeffs = []
for i in range(d1 + d2 + 1):
    resultingCoeffs.append(int(round(output.get_value_at_time(i))))
resultingCoeffs.reverse()

print(f"Degree of the Polynomial: {d1+d2}")
print(f"Coefficients: {' '.join(map(str, resultingCoeffs))}")

# Multiply the polynomials using Discrete-Time Convolution


# Print the result
