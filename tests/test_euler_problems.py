""" Unit tests for Euler problems

These tests check the result of Project Euler problems
"""
import pytest
from euler_problems import *
import assets


def test_problem_1():
    sum_multiples = problem_1(10)
    assert sum_multiples == 23


def test_problem_2():
    sum_fib = problem_2(4000)
    assert sum_fib == 3382


def test_problem_3():
    largest_prime_factor = problem_3(13195)
    assert largest_prime_factor == 29


def test_problem_4():
    largest_product = problem_4(100)
    assert largest_product == 9009


def test_problem_5():
    divisible = problem_5(10)
    assert divisible == 2520


def test_problem_6():
    sum_square_diff = problem_6(10)
    assert sum_square_diff == 2640


def test_problem_7():
    prime = problem_7(6)
    assert prime == 13


def test_problem_8():
    max_prod = problem_8(4)
    assert max_prod == 5832


def test_problem_9():
    triplet_product = problem_9(12)
    assert triplet_product == 60


def test_problem_10():
    prime_sum = problem_10(10)
    assert prime_sum == 17


def test_problem_11():
    grid = assets.PROBLEM_11_GRID
    biggest_product = problem_11(grid)
    assert biggest_product == 1788696


def test_problem_12():
    first_with_n_divs = problem_12(5)
    assert first_with_n_divs == 28


def test_problem_13():
    integer_list = assets.PROBLEM_13_INT_LIST
    first_10_sum = problem_13(integer_list)
    assert first_10_sum == 5537376230


def test_problem_14():
    longest_start = problem_14(100)
    assert longest_start == 97


def test_problem_15():
    n_routes = problem_15(2)
    assert n_routes == 6


def test_problem_16():
    digit_sum = problem_16(15)
    assert digit_sum == 26


def test_problem_17():
    letter_count = problem_17(5)
    assert letter_count == 19


def test_problem_20():
    factorial_digit_sum = problem_20(10)
    assert factorial_digit_sum == 27


def test_problem_21():
    amicable_sum = problem_21(10000)
    assert amicable_sum == 31627


def test_problem_23():
    sum_non_abundants = problem_23()
    assert sum_non_abundants == 123


def test_problem_24():
    nth_permuation = problem_24("012")
    assert nth_permuation == "120"


def test_problem_25():
    first_n_digit_index = problem_25(3)
    assert first_n_digit_index == 12


def test_problem_29():
    n_distinct_terms = problem_29(5)
    assert n_distinct_terms == 15


def test_problem_34():
    sum_curious_nums = problem_34()
    assert sum_curious_nums == 12487


def test_problem_39():
    n_solutions = problem_39(120)
    assert n_solutions == 120


def test_problem_47():
    consecutive_num = problem_47(2)
    assert consecutive_num == 14


def test_problem_48():
    sum_self_powers = problem_48(10)
    assert sum_self_powers == 10405071317


def test_problem_50():
    longest_sum_prime = problem_50(100)
    assert longest_sum_prime == 41
