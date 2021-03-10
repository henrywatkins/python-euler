""" Unit tests for Euler problems

These tests check the result of Project Euler problems
"""
import pytest
from euler_problems import *
import assets


def test_problem_13():
    integer_list = assets.PROBLEM_13_INT_LIST
    # integer_list = [23453892700, 49851123150, 19822383452]
    # short_answer = 23453892700 + 49851123150 + 19822383452
    # short_answer = int(str(short_answer)[:10])
    first_10_sum = problem_13(integer_list)
    assert first_10_sum == 5537376230


def test_problem_14():
    longest_start = problem_14(1000000)
    assert longest_start == 13
