""" Euler problems

Solutions to problems from Project Euler
"""
from typing import List, Tuple
from tqdm import tqdm
from math import sqrt, prod, factorial, ceil, floor, log10  # , gcd
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
    grid_size = len(grid)
    products = []
    for i in range(grid_size - 4):
        for j in range(grid_size - 4):
            h_prod = prod([grid[i][j], grid[i + 1][j], grid[i + 2][j], grid[i + 3][j]])
            v_prod = prod([grid[i][j], grid[i][j + 1], grid[i][j + 2], grid[i][j + 3]])
            dl_prod = prod(
                [grid[i][j], grid[i + 1][j + 1], grid[i + 2][j + 2], grid[i + 3][j + 3]]
            )
            dr_prod = prod(
                [grid[i][j + 3], grid[i + 1][j + 2], grid[i + 2][j + 1], grid[i + 3][j]]
            )
            products.extend([h_prod, v_prod, dl_prod, dr_prod])

    return max(products)


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


def problem_18(tri_grid: List[List[int]]) -> int:
    max_sum = 0
    for i in range(2 ** len(tri_grid)):
        sum = 0
        path = [int(i) for i in reversed(bin(i)[2:])]
        path.extend([0 for i in range(len(tri_grid) - len(path))])
        col = 0
        for row, step in zip(tri_grid, path):
            sum += row[col]
            col += step
        if sum > max_sum:
            max_sum = sum
    return max_sum


def problem_19() -> int:
    start, end = 1901, 2001
    day_of_week = 1
    n_sundays = 0
    for year in range(start, end):
        for month in range(1, 13):
            for day in range(1, get_month_days(month, year) + 1):
                day_of_week += 1
                if (day_of_week % 7 == 0) and (day == 1):
                    n_sundays += 1
    return n_sundays


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


def problem_22(filename: str) -> int:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_map = {char: i + 1 for i, char in enumerate(alphabet)}
    with open(filename, "r") as file:
        names_str = file.read()
    names_str = names_str.strip('"')
    split_names = names_str.split('","')
    sorted_names = sorted(split_names)
    name_scores = [
        (i + 1) * sum([char_map[c] for c in name])
        for i, name in enumerate(sorted_names)
    ]
    return sum(name_scores)


def problem_23() -> int:
    N_max = 281
    abundants = [i for i in range(1, N_max) if sum_proper_divisors(i) > i]
    combinations = set([i + j for i in abundants for j in abundants])
    not_abundant_sum = set(list(range(N_max))).difference(combinations)
    return sum(not_abundant_sum)


def problem_24(digits: List[int], n_perms: int) -> List[int]:
    for i in range(n_perms):
        lexi_permute(digits)
    return digits


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


def problem_26(max_n: int) -> int:
    longest_cycle_int, longest_cycle = 1, 0
    for i in range(1, max_n):
        remainders = []
        Q, R = 1 // i, 1 % i
        while R > 0:
            Rp = 10 * R
            Q, R = Rp // i, Rp % i
            if R in remainders:
                break
            remainders.append(R)
        if len(remainders) > longest_cycle:
            longest_cycle = len(remainders)
            longest_cycle_int = i
    return longest_cycle_int


def problem_27(max_ab: int) -> int:
    max_prime_prod = 0
    max_n_primes = 0
    for a in range(-max_ab, max_ab):
        for b in range(-max_ab, max_ab):
            n = 0
            current_n_primes = 0
            while len(prime_factors(n ** 2 + a * n + b)) == 1:
                n += 1
                current_n_primes += 1
            if current_n_primes > max_n_primes:
                max_n_primes = current_n_primes
                max_prime_prod = a * b
    return max_prime_prod


def problem_28(spiral_size: int) -> int:
    diag_sum = 1
    last_diag = 1
    layer_width = 3
    layer_max = layer_width ** 2
    while last_diag < spiral_size ** 2:
        last_diag += layer_width - 1
        diag_sum += last_diag
        if last_diag == layer_max:
            layer_width += 2
            layer_max = layer_width ** 2
    return diag_sum


def problem_29(x: int) -> int:
    terms = []
    for i in range(2, x + 1):
        for j in range(2, x + 1):
            terms.append(i ** j)
    return len(set(terms))


def problem_30(p: int) -> int:
    unique_power_sums = []
    vals = [i ** p for i in range(10)]
    max_digits = 4
    idx = [0 for i in range(max_digits)]
    while lexi_permute(idx, 9):
        power_sum = sum([vals[i] for i in idx])
        if power_sum == int("".join([str(j) for j in idx])):
            unique_power_sums.append(power_sum)
    return sum(unique_power_sums)


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
    MAX = 1000
    curious_nums = []
    for i in range(3, MAX):
        if sum([factorial(int(d)) for d in str(i)]) == i:
            curious_nums.append(i)
    return sum(curious_nums)


def problem_35(max_n: int) -> int:
    primes = primes_to(max_n)
    circular_primes = []
    for p in primes:
        Nd = floor(log10(p) + 1)
        str_p = str(p)
        rotations = [p]
        for i in range(Nd - 1):
            str_p = shift(1, str_p)
            rotations.append(int(str_p))
        prime_rotations = [r for r in rotations if len(prime_factors(r)) == 1]
        if len(rotations) == len(prime_rotations):
            circular_primes.append(p)
    n_circulars = len(set(circular_primes))
    return n_circulars


def problem_36(max_n: int) -> int:
    palindromes = []
    for i in range(max_n):
        binary = bin(i)[2:]
        is_bin_palindrome = list(reversed(binary)) == list(binary)
        is_dec_palindrome = list(reversed(str(i))) == list(str(i))
        if is_bin_palindrome and is_dec_palindrome:
            palindromes.append(i)
    return sum(palindromes)


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


def problem_40(p: int) -> int:
    max = 10 ** p
    all_ints = "".join([str(i) for i in range(1, max)])
    indices = [10 ** i for i in range(p)]
    digits = [int(all_ints[i - 1]) for i in indices]
    return prod(digits)


def problem_45(n: int) -> int:
    pass


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


def problem_52(max_mult: int) -> int:
    x = 1
    while True:
        x_digits = sorted(str(x))
        if all([sorted(str(m * x)) == x_digits for m in range(2, max_mult + 1)]):
            return x
        x += 1


def problem_56(max: int) -> int:
    largest = 0
    for i in range(max):
        for j in range(max):
            exponent = binary_exponent(i, j)
            digits = list(str(exponent))
            sum_digits = sum([int(i) for i in digits])
            if sum_digits > largest:
                largest = sum_digits
    return largest


def problem_57(max_exp: int) -> int:
    n_more_numerators = 0
    h0, k0 = 3, 2
    h1, k1 = 7, 5
    dummy = 0
    a = 2
    for i in range(2, max_exp):
        dummy = h1
        h1 = 2 * h1 + h0
        h0 = dummy
        dummy = k1
        k1 = 2 * k1 + k0
        k0 = dummy
        if len(str(h1)) > len(str(k1)):
            n_more_numerators += 1
    return n_more_numerators


def problem_63(max_p: int) -> int:
    counts = 0
    for p in range(1, max_p):
        for x in range(1, 11):
            if (p - 1) == floor(p * log10(x)):
                counts += 1
    return counts


def problem_92(max_n: int) -> int:
    n_to_89 = 0
    for i in range(1, max_n):
        chain = i
        while True:
            chain = sum([int(j) ** 2 for j in str(chain)])
            if chain == 89:
                n_to_89 += 1
                break
            elif chain == 1:
                break
            else:
                continue
    return n_to_89
