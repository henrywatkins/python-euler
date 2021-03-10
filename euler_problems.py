""" Euler problems

Solutions to problems from Project Euler
"""
import pytest
from typing import List, Tuple
from tqdm import tqdm
from math import sqrt, prod


def prime_factors(n: int) -> List[int]:
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


def collatz_seq_len(start: int) -> int:
    current_num = start
    seq_length = 1
    while current_num > 1:
        if current_num % 2 == 0:
            current_num = int(current_num / 2)
        else:
            current_num = 3 * current_num + 1
        seq_length += 1
    return seq_length


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


def problem_16(power: int) -> int:
    digit_string = str(2 ** power)
    digit_sum = sum([int(i) for i in digit_string])
    return digit_sum


DIGIT_NAMES = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
TEEN_NAMES = {
    "0": "ten",
    "1": "eleven",
    "2": "twelve",
    "3": "thirteen",
    "4": "fourteen",
    "5": "fifteen",
    "6": "sixteen",
    "7": "seventeen",
    "8": "eighteen",
    "9": "nineteen",
}
DECILE_NAMES = {
    "2": "twenty",
    "3": "thirty",
    "4": "forty",
    "5": "fifty",
    "6": "sixty",
    "7": "seventy",
    "8": "eighty",
    "9": "ninety",
}


def get_single_fig_name(string: str) -> str:
    return DIGIT_NAMES[string]


def get_double_fig_name(string_1: str, string_2: str) -> str:
    if string_1 == "1":
        number_string = TEEN_NAMES[string_2]
    else:
        if string_2 == "0":
            number_string = DECILE_NAMES[string_1]
        else:
            number_string = DECILE_NAMES[string_1] + " " + get_single_fig_name(string_2)
    return number_string


def get_triple_fig_name(string_1: str, string_2: str, string_3: str) -> str:
    if string_2 == "0":
        if string_3 == "0":
            number_string = DIGIT_NAMES[string_1] + " hundred"
        else:
            number_string = (
                DIGIT_NAMES[string_1] + " hundred and " + get_single_fig_name(string_3)
            )
    else:
        number_string = (
            DIGIT_NAMES[string_1]
            + " hundred and "
            + get_double_fig_name(string_2, string_3)
        )
    return number_string


def number_to_string(num: int) -> str:
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


def problem_17(max_n: int) -> int:
    letter_string = "".join([number_to_string(i) for i in range(1, max_n + 1)])
    letter_string_no_spc = letter_string.replace(" ", "")
    letter_count = len(letter_string_no_spc)
    return letter_count
