import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, lambdify


def user_function(vetor_da_variaveis):
    global funcao
    # Cria uma expressão simbólica da função a partir da string fornecida pelo usuário
    new_function = sympify(funcao)
    new_function = new_function.subs(x, vetor_da_variaveis[0]).subs(y, vetor_da_variaveis[1])
    return new_function


def newton_modificado(v0, epsilon, user_function):
    global funcao
    x0, y0 = symbols('x y')

    max_iter = 1000
    num_iters = []
    x_vals = []
    y_vals = []

    # Calcula o gradiente da função
    grad_x = diff(funcao, x).subs(x, v0[0]).subs(y, v0[1])
    grad_y = diff(funcao, y).subs(x, v0[0]).subs(y, v0[1])
    grad = [grad_x, grad_y]

    # Calcule as derivadas parciais de segunda ordem
    d2f_dx2 = diff(diff(funcao, x), x)
    d2f_dy2 = diff(diff(funcao, y), y)
    d2f_dxdy = diff(diff(funcao, x), y)

    # Converta as expressões simbólicas em funções numéricas
    d2f_dx2_func = lambdify((x, y), d2f_dx2)
    d2f_dy2_func = lambdify((x, y), d2f_dy2)
    d2f_dxdy_func = lambdify((x, y), d2f_dxdy)

    # Calcule os valores numéricos das derivadas parciais de segunda ordem
    d2f_dx2_val = d2f_dx2_func(v0[0], v0[1])
    d2f_dy2_val = d2f_dy2_func(v0[0], v0[1])
    d2f_dxdy_val = d2f_dxdy_func(v0[0], v0[1])

    # Crie a matriz Hessiana
    hessiana = np.array([[d2f_dx2_val, d2f_dxdy_val], [d2f_dxdy_val, d2f_dy2_val]], dtype=np.float64)

    # Calcule a inversa da Hessiana
    hessiana_inversa = np.linalg.inv(hessiana)

    dk = np.dot(hessiana_inversa, grad)

    """for i in range(max_iter):
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
        y_vals.append(x[1])"""

    return x, user_function(x), 1


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
