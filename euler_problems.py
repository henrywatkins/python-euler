""" Euler problems

Solutions to problems from Project Euler
"""
import pytest
from typing import List, Tuple
from tqdm import tqdm


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
    # remember seen numbers, as they have already been checked
    current_int, longest_seq = 1, 0
    current_best = current_int
    while current_int < max_number:
        length = collatz_seq_len(current_int)
        if length > longest_seq:
            longest_seq = length
            current_best = current_int
        current_int += 1
    return current_best
