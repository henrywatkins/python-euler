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
    assert biggest_product == 70600674


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


def test_problem_18():
    tri_values = assets.PROBLEM_18_TRI
    max_sum_path = problem_18(tri_values)
    assert max_sum_path == 23


def test_problem_19():
    n_sundays = problem_19()
    assert n_sundays == 171


def test_problem_20():
    factorial_digit_sum = problem_20(10)
    assert factorial_digit_sum == 27


def test_problem_21():
    amicable_sum = problem_21(10000)
    assert amicable_sum == 31627


def test_problem_22():
    filename = "./assets/names.txt"
    names_scores = problem_22(filename)
    assert names_scores == 871198282


def test_problem_23():
    sum_non_abundants = problem_23()
    assert sum_non_abundants == 19866


def test_problem_24():
    n = 5
    nth_permuation = problem_24([0, 1, 2], n)
    assert nth_permuation == [2, 1, 0]


def test_problem_25():
    first_n_digit_index = problem_25(3)
    assert first_n_digit_index == 12


def test_problem_26():
    longest_cycle = problem_26(10)
    assert longest_cycle == 7


def test_problem_27():
    max_sequence_prod = problem_27(50)
    assert max_sequence_prod == -235


def test_problem_28():
    spiral_diag_sum = problem_28(5)
    assert spiral_diag_sum == 101


def test_problem_29():
    n_distinct_terms = problem_29(5)
    assert n_distinct_terms == 15


def test_problem_30():
    sum_n_powers = problem_30(4)
    assert sum_n_powers == 19316


def test_problem_31():
    count_ways = problem_31()
    assert count_ways == 73682


def test_problem_32():
    pandigital_product_sum = problem_32(4)
    assert pandigital_product_sum == 12


def test_problem_33():
    curious_fraction_product = problem_33()
    assert curious_fraction_product == 100


def test_problem_34():
    sum_curious_nums = problem_34()
    assert sum_curious_nums == 145


def test_problem_35():
    n_circular_primes = problem_35(100)
    assert n_circular_primes == 13


def test_problem_36():
    max_n = 10
    palindrome_sum = problem_36(max_n)
    assert palindrome_sum == 25


def test_problem_37():
    max_n = 10000
    trunc_prime_sum = problem_37(max_n)
    assert trunc_prime_sum == 8920


def test_problem_38():
    max = 4  # 10
    largest_pandigital = problem_38(max)
    assert largest_pandigital == 321  # 932718654


def test_problem_39():
    n_solutions = problem_39(120)
    assert n_solutions == 120


def test_problem_40():
    champerdowne_digit_product = problem_40(1)
    assert champerdowne_digit_product == 1


def test_problem_41():
    max_n = 5  # 10
    biggest_pandigital_prime = problem_41(max_n)
    assert biggest_pandigital_prime == 0  # 7652413


def test_problem_42():
    n_triangle_words = problem_42()
    assert n_triangle_words == 162


def test_problem_43():
    max_digit = 4  # 10
    pandigital_property_sum = problem_43(max_digit)
    assert pandigital_property_sum == 22212  # 16695334890


def test_problem_44():
    max_N = 100  # 10000
    min_pentagonal_difference = problem_44(max_N)
    assert min_pentagonal_difference == 100000000  # 5482660


def test_problem_45():
    max_n = 1000000
    tri_pent_hex = problem_45(max_n)
    assert tri_pent_hex == 1533776805


def test_problem_46():
    smallest_odd_composite = problem_46()
    assert smallest_odd_composite == 5777


def test_problem_47():
    consecutive_num = problem_47(2)
    assert consecutive_num == 14


def test_problem_48():
    sum_self_powers = problem_48(10)
    assert sum_self_powers == 10405071317


def test_problem_49():
    digit_permuatation = problem_49()
    assert digit_permuatation == 296962999629


def test_problem_50():
    max_n = 1000  # 1000000
    longest_sum_prime = problem_50(max_n)
    assert longest_sum_prime == 953  # 997651


def test_problem_52():
    smallest_permuted_multiple = problem_52(2)
    assert smallest_permuted_multiple == 125874


def test_problem_53():
    N = 100
    n_more_than_mil = problem_53(N)
    assert n_more_than_mil == 4


def test_problem_55():
    max_n = 10000
    n_lychrel_numbers = problem_55(max_n)
    assert n_lychrel_numbers == 120


def test_problem_56():
    max_digit_sum = problem_56(3)
    assert max_digit_sum == 4


def test_problem_57():
    n_bigger_numerators = problem_57(9)
    assert n_bigger_numerators == 1


def test_problem_58():
    thresh = 0.4  # 0.1
    side_length = problem_58(thresh)
    assert side_length == 31  # 26241


def test_problem_59():
    ascii_value_sum = problem_59()
    assert ascii_value_sum == 129448


def test_problem_63():
    n_digit_n_power_count = problem_63(100)
    assert n_digit_n_power_count == 49


def test_problem_64():
    N = 100
    n_odd_periods = problem_64(N)
    assert n_odd_periods == 4


def test_problem_69():
    N = 10
    euler_totient_max = problem_69(N)
    assert euler_totient_max == 6


def test_problem_70():
    N = 100
    euler_totient_minimum = problem_70(N)
    assert euler_totient_minimum == 7


def test_problem_74():
    max = 1000000
    n_chains_60_terms = problem_74(max)
    assert n_chains_60_terms == 23

def test_problem_79():
    passcode = problem_79()
    assert passcode==124988

def test_problem_92():
    n_below = problem_92(100)
    assert n_below == 80


def test_problem_97():
    last_ten_digits = problem_97()
    assert last_ten_digits == 1907825665

def test_problem_99():
    largest_value_line = problem_99()
    assert largest_value_line==708
