""" Euler problems

Solutions to problems from Project Euler
"""
from typing import List, Tuple
from tqdm import tqdm
from math import sqrt, prod, factorial, ceil, floor  # , gcd
import numpy as np
import assets
from utils import *


def problem_1(max_n: int) -> int:
    multiples = [i for i in range(1, max_n) if (i % 3 == 0) or (i % 5 == 0)]
    return sum(multiples)


def problem_2(max_n: int) -> int:
    sum = 0
    current = 1
    previous = 0
    dummy = 0
    while current < max_n:
        dummy = current
        current += previous
        previous = dummy
        if current % 2 == 0:
            sum += current
    return sum


def problem_3(n: int) -> int:
    factors = prime_factors(n)
    return max(factors)


def problem_4(max_n: int) -> int:
    best = 0
    for i in range(1, max_n):
        for j in range(1, max_n):
            candidate_string = str(j * i)
            reversed = list(candidate_string)
            reversed.reverse()
            if all([a == b for a, b in zip(candidate_string, reversed)]):
                if j * i > best:
                    best = j * i
    return best


def problem_5(n: int) -> int:
    factors = []
    for div in range(1, n + 1):
        divisor_factors = prime_factors(div)
        for div_fac in set(divisor_factors):
            div_fac_pow = divisor_factors.count(div_fac)
            fac_pow = factors.count(div_fac)
            if fac_pow < div_fac_pow:
                factors.extend([div_fac for i in range(div_fac_pow - fac_pow)])
    start = prod(factors)
    return start


def problem_6(n: int) -> int:
    square_sum = sum(range(1, n + 1)) ** 2
    sum_square = sum([i ** 2 for i in range(1, n + 1)])
    return square_sum - sum_square


def problem_7(n: int) -> int:
    primes = primes_to(1100)
    nth_prime = primes[n - 1]
    return nth_prime


def problem_8(n: int) -> int:
    int_string = assets.PROBLEM_8_STRING
    digits = [int(i) for i in int_string]
    best = 0
    for i in range(len(digits) - n):
        current = prod(digits[i : i + n])
        if current > best:
            best = current
    return best


def problem_9(triplet_sum: int) -> int:
    solutions = triangle_sols(triplet_sum)
    k, m, n = solutions[0]
    a, b, c = euclid_formula(k, m, n)
    triplet_product = int(a * b * c)
    return triplet_product


def problem_10(n: int) -> int:
    primes = primes_to(n)
    return sum(primes)


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
    adjacency_matrix = grid_to_matrix(grid_size)
    dummy_adj = grid_to_matrix(grid_size)
    path_length = 2 * grid_size
    adjacency_matrix = np.linalg.matrix_power(adjacency_matrix, path_length)
    n_paths = adjacency_matrix[0, -1]
    return n_paths


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
        sum_divs = sum_proper_divisors(i)
        to_check = sum_proper_divisors(sum_divs)
        if (to_check == i) and (sum_divs != i):
            amicable_nums.append(i)
    amicable_sum = sum(amicable_nums)
    return amicable_sum


def problem_23() -> int:

    return 0


def permute(vals: List[int]) -> List[int]:
    # """permute a list of ints based on lexicographic order"""
    k = len(vals)
    n = max(vals)
    for i in range(k - 1, -1, -1):
        if vals[i] < n - k + i + 1:
            vals[i] += 1
            for j in range(i + 1, k):
                vals[j] = vals[j - 1] + 1
    return vals


def problem_24(digits: str) -> str:
    str_sort = sorted(digits)
    return


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


def problem_29(x: int) -> int:
    terms = []
    for i in range(2, x + 1):
        for j in range(2, x + 1):
            terms.append(i ** j)
    return len(set(terms))


class DigitFactorializer:
    """calculate sum of factorials of the digits of an integer"""

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
    while n < 1:
        n += 1
        factorial_sum = factorializer(n)
        if n == factorial_sum:
            curious_nums.append(n)
    sum_curious_nums = sum(curious_nums)
    return sum_curious_nums


def problem_39(max_p: int) -> int:
    max_sols = 0
    best_p = 1
    for i in range(1, max_p + 1):
        sols = triangle_sols(i)
        triplets = [sorted(euclid_formula(*sol)) for sol in sols]
        unique_triplets = [list(x) for x in set(tuple(x) for x in triplets)]
        n_sols = len(unique_triplets)
        if n_sols > max_sols:
            max_sols = n_sols
            best_p = i

    return best_p


def problem_47(n: int) -> int:

    consecutive = [i for i in range(1, n + 1)]
    while True:
        distinct_factors = [set(prime_factors(i)) for i in consecutive]
        if all([len(factors) == n for factors in distinct_factors]):
            break
        consecutive = [i + 1 for i in consecutive]

    return consecutive[0]


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
