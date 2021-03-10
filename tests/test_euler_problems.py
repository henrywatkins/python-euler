""" Unit tests for Euler problems

These tests check the result of Project Euler problems
"""
import pytest
from euler_problems import *
import assets


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


def test_problem_16():
    digit_sum = problem_16(15)
    assert digit_sum == 26


def test_problem_17():
    letter_count = problem_17(5)
    assert letter_count == 19
