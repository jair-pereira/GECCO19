from math import *
import numpy as np

def ackley_function(x):
    return -20*exp(- 0.2 * sqrt(sum([i**2 for i in x])/len(x))) - exp(sum([cos(2*pi*i) for i in x])/ len(x)) + 20 + exp(1)


def bukin_function(x):
    return 100*sqrt(abs(x[1]-0.01*x[0]**2)) + 0.01*abs(x[0] + 10)


def cross_in_tray_function(x):
    return round(-0.0001*(abs(sin(x[0])*sin(x[1])*exp(abs(100 -
                            sqrt(sum([i**2 for i in x]))/pi))) + 1)**0.1, 7)


def sphere_function(x):
    return sum([i**2 for i in x])


def bohachevsky_function(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*cos(3*pi*x[0]) - 0.4*cos(4*pi*x[1]) + 0.7


def sum_squares_function(x):
    return sum([(i+1)*x[i]**2 for i in range(len(x))])


def sum_of_different_powers_function(x):
    return sum([abs(x[i])**(i+2) for i in range(len(x))])


def booth_function(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas_function(x):
    return 0.26*sphere_function(x) - 0.48*x[0]*x[1]


def mccormick_function(x):
    return sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


def dixon_price_function(x):
    return (x[0] - 1)**2 + sum([(i+1)*(2*x[i]**2 - x[i-1])**2
                                for i in range(1, len(x))])


def six_hump_camel_function(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1]\
           + (-4 + 4*x[1]**2)*x[1]**2


def three_hump_camel_function(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


def easom_function(x):
    return -cos(x[0])*cos(x[1])*exp(-(x[0] - pi)**2 - (x[1] - pi)**2)


def michalewicz_function(x):
    return -sum([sin(x[i])*sin((i+1)*x[i]**2/pi)**20 for i in range(len(x))])


def beale_function(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + \
           (2.625 - x[0] + x[0]*x[1]**3)**2


def drop_wave_function(x):
    return -(1 + cos(12*sqrt(sphere_function(x))))/(0.5*sphere_function(x) + 2)

def katsuura(x):
    def coco_double_round(number):
        return floor(number + 0.5)

    # translated from COCO
    number_of_variables = len(x)

    result = 1.0
    for i in range(number_of_variables):
        tmp = 0
        for j in range(1, 33):
            tmp2 = 2.**j
            tmp += fabs(tmp2 * x[i] - coco_double_round(tmp2 * x[i])) / tmp2

        tmp = 1.0 + (i + 1) * tmp
        result *= tmp

    result = 10. / (number_of_variables) / (number_of_variables) * (-1. + result**(10. / number_of_variables**1.2))

    return result