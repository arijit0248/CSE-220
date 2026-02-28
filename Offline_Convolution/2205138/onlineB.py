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


# # Stock Market Prices as a Python List
# price_list = list(map(int, input("Stock Prices: ").split()))
# n = int(input("Window size: "))
price_list = [1, 2, 3, 4, 5, 6, 7, 8]
n = 4

INF_x = len(price_list)
x = Signal(INF_x)
for i in range(INF_x):
    x.set_value_at_time(i, price_list[i])

INF_h = n
h1 = Signal(INF_h)
for i in range(n):
    h1.set_value_at_time(i, (1/n))
temp = (n*(n+1)) / 2
h2 = Signal(INF_h)
for i in range(n):
    coefficient = (n-i)/temp
    h2.set_value_at_time(i, coefficient)
# Please determine uma and wma.
lti1 = LTI_System(h1)
lti2 = LTI_System(h2)
y1 = lti1.output(x)
y2 = lti2.output(x)
# Unweighted Moving Averages as a Python list
uma = []
for i in range(n-1, len(price_list)):
    uma.append(y1.get_value_at_time(i))

# Weighted Moving Averages as a Python list
wma = []
for i in range(n-1, len(price_list)):
    wma.append(y2.get_value_at_time(i))

# Print the two moving averages
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))
