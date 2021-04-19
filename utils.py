""" Utility functions for Euler problems
"""
from math import sqrt, prod, factorial, ceil, floor
from typing import List, Tuple
import assets
import numpy as np


def prime_factors(n: int) -> List[int]:
    """Calculate prime factors of integer"""
    factorization = []
    i = 2
    while i * i <= n:
        while n % i == 0:
            factorization.append(i)
            n //= i
        i += 1

    if n > 1:
        factorization.append(n)

    return factorization


def gcd(a: int, b: int) -> int:
    """implementation of euclid's algorithm
    to find greatest common divisor of two integers"""
    dummy = 0
    while b:
        a %= b
        x = b
        b = a
        a = x
    return a


def euclid_formula(k: int, m: int, n: int) -> Tuple[int]:
    """implementation of euclid's formula to find
    pythagorean_triplets"""
    return k * (m ** 2 - n ** 2), k * 2 * m * n, k * (m ** 2 + n ** 2)


def collatz_seq_len(start: int) -> int:
    """calculate the length of the Collatz sequence starting at integer"""
    current_num = start
    seq_length = 1
    while current_num > 1:
        if current_num % 2 == 0:
            current_num = int(current_num / 2)
        else:
            current_num = 3 * current_num + 1
        seq_length += 1
    return seq_length


def euler_totient(n: int) -> int:
    """ calcualte the euler totient function of an integer"""
    factors = prime_factors(n)
    to_product = [(1 - 1 / p) for p in set(factors)]
    product = prod(to_product)
    return int(n * product)


def num_divisors(n: int) -> int:
    """calculate the number of divisors of an integer"""
    factors = prime_factors(n)
    to_product = [factors.count(p) + 1 for p in set(factors)]
    product = prod(to_product)
    return int(product)


def sum_divisors(n: int) -> int:
    """sum divisors of integer"""
    factors = prime_factors(n)
    to_product = [(p ** (factors.count(p) + 1) - 1) / (p - 1) for p in set(factors)]
    sum_divisors = int(prod(to_product))
    return sum_divisors


def sum_proper_divisors(n: int) -> int:
    """sum the proper divisors of integer"""
    sum_divs = sum_divisors(n) - n
    return sum_divs


def binary_exponent(a: int, b: int) -> int:
    """raise a to power of b using binary exponentiation"""
    answer = 1
    while b > 0:
        if b & 1:
            answer *= a
        a *= a
        b >>= 1
    return answer


def self_power(x: int) -> int:
    """raise x to the power of itself using binary exponentiation"""
    answer = binary_exponent(x, x)
    return answer


def primes_to(n: int) -> List[int]:
    """Use sieve of eratosthenes to find all primes numbers below n"""
    is_prime = [True for i in range(n + 1)]
    is_prime[0], is_prime[1] = False, False
    for i in range(2, n + 1):
        if is_prime[i] and i * i <= n:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]


def triangle_sols(p: int) -> List[Tuple[int]]:
    """find pythagorean triplet solutions for
    triangle of perimeter p"""
    solutions = []
    kmax = ceil(p / 2)
    for k in range(1, kmax):
        mmax = ceil(sqrt(p / (2 * k)))
        for m in range(1, mmax):
            for n in range(1, m):
                if 2 * m * k * (m + n) == p:
                    solutions.append((k, m, n))
    return solutions


def grid_to_matrix(grid_size: int):
    vertex_size = grid_size + 1
    n_vertices = vertex_size ** 2
    adjacency_matrix = np.matrix(np.zeros((n_vertices, n_vertices)), dtype=np.int64)
    for i in range(vertex_size):
        for j in range(vertex_size):
            index = j + vertex_size * i
            if i < vertex_size - 1:
                adjacency_matrix[index, index + vertex_size] = 1
            if j < vertex_size - 1:
                adjacency_matrix[index, index + 1] = 1
    return adjacency_matrix


def number_chain_step(x: int) -> int:
    """add square of digits in integer to self"""
    digits = list(str(x))
    sum_sqr = sum([int(i) ** 2 for i in digits])
    return sum_sqr


def matrix_sum(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    assert len(A) == len(B)
    assert all([len(i) == len(j) for i, j in zip(A, B)])
    output = [[i + j for i, j in zip(a, b)] for a, b in zip(A, B)]
    return output


def matrix_product(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    assert all([len(i) == len(B) for i in A])
    output = [
        [
            sum([i * j for i, j in zip(a, [B[k][b] for k in range(len(B))])])
            for b in range(len(B[0]))
        ]
        for a in A
    ]
    return output


def get_single_fig_name(string: str) -> str:
    return assets.DIGIT_NAMES[string]


def get_double_fig_name(string_1: str, string_2: str) -> str:
    if string_1 == "1":
        number_string = assets.TEEN_NAMES[string_2]
    else:
        if string_2 == "0":
            number_string = assets.DECILE_NAMES[string_1]
        else:
            number_string = (
                assets.DECILE_NAMES[string_1] + " " + get_single_fig_name(string_2)
            )
    return number_string


def get_triple_fig_name(string_1: str, string_2: str, string_3: str) -> str:
    if string_2 == "0":
        if string_3 == "0":
            number_string = asssets.DIGIT_NAMES[string_1] + " hundred"
        else:
            number_string = (
                assets.DIGIT_NAMES[string_1]
                + " hundred and "
                + get_single_fig_name(string_3)
            )
    else:
        number_string = (
            assets.DIGIT_NAMES[string_1]
            + " hundred and "
            + get_double_fig_name(string_2, string_3)
        )
    return number_string


def number_to_string(num: int) -> str:
    """convert integer to a named string"""
    digits = list(str(num))
    digit_length = len(digits)
    if digit_length == 4:
        number_string = "one thousand"
    elif digit_length == 3:
        number_string = get_triple_fig_name(digits[0], digits[1], digits[2])
    elif digit_length == 2:
        number_string = get_double_fig_name(digits[0], digits[1])
    elif digit_length == 1:
        number_string = get_single_fig_name(digits[0])
    else:
        number_string = ""
    return number_string
