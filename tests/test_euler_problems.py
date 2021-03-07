""" Unit tests for Euler problems

These tests check the result of Project Euler problems
"""
import pytest
from euler_problems import *


def test_problem_13():
    integer_list = [23453892700, 49851123150, 19822383452]
    first_10_sum = problem_13(integer_list)
    assert first_10_sum==sum(tuple(integer_list))
