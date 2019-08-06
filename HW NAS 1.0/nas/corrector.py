import numpy as np
import matplotlib.pyplot as plt

# from .evaluater import Evaluater


class EvalCorrector():
    def __init__(self):
        self.beta1 = 1
        self.beta2 = 0
        self.eta = 0.1
        self.a = 1
        self.b = 1/3
        self.D = []
        return

    def update(self, x, y):
        self.D.append((x, y))
        if len(self.D) == 0:
            self.beta1 = pow(y, 1 / self.b) / pow(self.a, 1 / self.b) / x
        else:
            p1 = 0
            for i in range(len(self.D)):
                p1 += 2 * (self.func(self.D[i][0]) - self.D[i][1]) * (
                        self.D[i][0] * self.b * self.func(self.D[i][0]) / (self.beta1 * self.D[i][0] + self.beta2))
            p2 = 0
            for i in range(len(self.D)):
                p2 += 2 * (self.func(self.D[i][0]) - self.D[i][1]) * (
                        self.b * self.func(self.D[i][0]) / (self.beta1 * self.D[i][0] + self.beta2))
            self.beta2 -= self.eta * p2
            self.beta1 -= self.eta * p1
        return

    def corrector(self, x):
        return self.func(x)

    def func(self, x):
        # y = 1
        # y = pow((self.beta1 * x + self.beta2),self.b)
        y = pow(x, self.b)
        y *= self.a
        return y


def fun1(x):
    step=0.1
    y=0
    for i in range(10):
        y+=pow(x,i/40)*i/65
    return y


if __name__ == '__main__':
    c = EvalCorrector()
    y = [0.447, 0.511, 0.506, 0.554, 0.567, 0.595, 0.594, 0.587, 0.597, 0.587, 0.577, 0.593, 0.586, 0.582, 0.586, 0.596,
         0.602, 0.606]
    for i in range(18):
        c.update(0.05 * (i + 1), y[i])
        print(c.corrector(0.05 * (i + 1)))

    x=[0.05 * (i + 1) for i in range(18)]
    plt.plot(x,[fun1(i)for i in x])
    plt.scatter(x,y)
    plt.show()
