import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify


def user_function(vetor_da_variaveis):
    global funcao
    # Cria uma expressão simbólica da função a partir da string fornecida pelo usuário
    new_function = sympify(funcao)
    new_function = new_function.subs(x, vetor_da_variaveis[0]).subs(y, vetor_da_variaveis[1])
    return new_function


def newton_modificado(v0, epsilon, user_function):
    x = v0
    n = len(x)
    B = np.eye(n)  # initial approximation of inverse Hessian
    grad = lambda x: np.array([4 * x[0] ** 3 - 4.2 * x[0] ** 2 + x[1], x[0] + 2 * x[1]])  # gradient of three-hump camel
    max_iter = 1000
    num_iters = []
    x_vals = []
    y_vals = []

    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < epsilon:
            break  # stopping criterion for the gradient
        p = -np.dot(B, g)
        alpha = 1
        print(x + alpha * p)
        while user_function(x + alpha * p) > user_function(x) + 0.1 * alpha * np.dot(g, p):
            alpha *= 0.5
        s = alpha * p
        x_new = x + s
        y = grad(x_new) - g
        dot_product = np.dot(y, s)
        if dot_product == 0:
            rho = 0  # or any other appropriate value
        else:
            rho = 1 / dot_product
        B = (np.eye(n) - rho * np.outer(s, y)).dot(B).dot(np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        if np.linalg.norm(s) < epsilon:
            break  # stopping criterion for the variables
        num_iters.append(i + 1)
        x_vals.append(x[0])
        y_vals.append(x[1])

    return x, user_function(x), i


# Define os símbolos das variáveis independentes
x, y = symbols('x y')

# Entrada das informações do usuário
funcao = input("Insira a função que deseja otimizar: ")

# Solicita os valores das variáveis x e y
x0 = float(input("Insira o x inicial: "))
y0 = float(input("Insira o y inicial: "))

# A variável inicial recebe os valores do usuário
v0 = np.array([x0, y0])

epsilon = 1e-6

# Substitui as variáveis digitadas na função
user_function(v0)

# Chama o método de Newton Modificado
x_opt, f_opt, num_iter = newton_modificado(v0, epsilon, user_function)

print(f"Solução ótima encontrada: x = {x_opt}, f(x) = {f_opt}, número de interações = {num_iter}")
