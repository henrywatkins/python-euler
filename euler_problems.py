""" Euler problems

Solutions to problems from Project Euler
"""
from typing import List, Tuple
from tqdm import tqdm
from math import sqrt, prod, factorial
import numpy as np
import assets
from utils import *


def problem_11(grid: List[List[int]]) -> int:
    step = 4
    horizontal_products = [
        [prod(row[i : i + 4]) for i in range(0, len(row), step)] for row in grid
    ]
    transpose_grid = [[row[i] for row in grid] for i in range(len(grid))]
    vertical_products = [
        [prod(row[i : i + 4]) for i in range(0, len(row), step)]
        for row in transpose_grid
    ]
    # left_diagonal_products =
    # right_diagonal_products =
    all_products = []
    all_products.extend([i for row in horizontal_products for i in row])
    all_products.extend([i for row in vertical_products for i in row])
    biggest_product = max(all_products)
    return biggest_product


def problem_12(max_divisors: int) -> int:
    n = 1
    n_divisors = 0
    while n_divisors <= max_divisors:
        tri_n = n * (n + 1) // 2
        factors = prime_factors(tri_n)
        count_unique = [factors.count(i) + 1 for i in set(factors)]
        n_divisors = prod(count_unique)
        n += 1
    return tri_n


def problem_13(int_series: List[int]) -> int:
    integer_sum = sum(int_series)
    integer_string = str(integer_sum)
    first_10 = integer_string[:10]
    first_10_int = int(first_10)
    return first_10_int


def problem_14(max_number: int) -> int:
    # could memoizaition, dynamical algorithm, but brute force for now
    current_int, longest_seq = 1, 0
    current_best = current_int
    while current_int < max_number:
        length = collatz_seq_len(current_int)
        if length > longest_seq:
            longest_seq = length
            current_best = current_int
        current_int += 1
    return current_best


def problem_15(grid_size: int) -> int:
    # n_vertices = (grid_size+1)**2
    # adjacency_matrix = np.matrix(np.zeros((n_vertices, n_vertices)), dtype=np.int)
    # for i in range(n_vertices):
    # path_length = 2*(grid_size+1)
    # matrix_product = np.linalg.matrix_power(adjacency_matrix, path_length)
    # return n_routes
    pass


def problem_16(power: int) -> int:
    digit_string = str(2 ** power)
    digit_sum = sum([int(i) for i in digit_string])
    return digit_sum


def problem_17(max_n: int) -> int:
    letter_string = "".join([number_to_string(i) for i in range(1, max_n + 1)])
    letter_string_no_spc = letter_string.replace(" ", "")
    letter_count = len(letter_string_no_spc)
    return letter_count


def problem_20(n: int) -> int:
    ftrl = factorial(n)
    digits = list(str(ftrl))
    digit_sum = sum([int(i) for i in digits])
    return digit_sum


def problem_21(max_n: int) -> int:
    amicable_nums = []
    for i in range(1, max_n):
        sum_divisors = sum_proper_divisors(i)
        to_check = sum_proper_divisors(sum_divisors)
        if (to_check == i) and (sum_divisors != i):
            amicable_nums.append(i)
    amicable_sum = sum(amicable_nums)
    return amicable_sum


def problem_25(n_digits: int) -> int:
    fn1, fn2 = 1, 0
    fn = fn1 + fn2
    index = 2
    while len(str(fn)) < n_digits:
        fn2 = fn1
        fn1 = fn
        fn = fn1 + fn2
        index += 1
    return index


class DigitFactorializer:
    def __init__(self):
        self.digit_mapping = {i: factorial(int(i)) for i in list("0123456789")}

    def __call__(self, x: int) -> int:
        digits = list(str(x))
        ftrl = [self.digit_mapping[i] for i in digits]
        factorial_sum = sum(ftrl)
        return factorial_sum


def problem_34() -> int:
    factorializer = DigitFactorializer()
    curious_nums = []
    n = 2
    factorial_sum = 2
    while n <= factorial_sum:
        n += 1
        factorial_sum = factorializer(n)
        print(n)
        print(factorial_sum)
        if n == factorial_sum:
            curious_nums.append(n)
    sum_curious_nums = sum(curious_nums)
    return sum_curious_nums


def problem_48(n: int) -> int:
    self_power_sum = sum([self_power(i) for i in range(1, n + 1)])
    return self_power_sum


def problem_50(max_n: int) -> int:
    primes_below = primes_to(max_n - 1)
    rolling_sum = []
    last_val = 0
    for i in primes_below:
        last_val += i
        if last_val in primes_below:
            rolling_sum.append(last_val)
    best_prime = max(rolling_sum)
    return best_prime
