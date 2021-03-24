""" Unit tests for Euler problems

These tests check the result of Project Euler problems
"""
import pytest
from euler_problems import *
import assets


def test_problem_11():
    grid = assets.PROBLEM_11_GRID
    biggest_product = problem_11(grid)
    assert biggest_product == 1788696


def test_problem_12():
    first_with_n_divs = problem_12(5)
    assert first_with_n_divs == 28


def test_problem_13():
    integer_list = assets.PROBLEM_13_INT_LIST
    # integer_list = [23453892700, 49851123150, 19822383452]
    # short_answer = 23453892700 + 49851123150 + 19822383452
    # short_answer = int(str(short_answer)[:10])
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


def test_problem_25():
    first_n_digit_index = problem_25(3)
    assert first_n_digit_index == 12


def test_problem_34():
    sum_curious_nums = problem_34()
    assert sum_curious_nums == 12487


def test_problem_48():
    sum_self_powers = problem_48(10)
    assert sum_self_powers == 10405071317


def test_problem_50():
    longest_sum_prime = problem_50(100)
    assert longest_sum_prime == 41
