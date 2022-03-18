# Euler Problems

This repository contains solutions to problems from the Project Euler website www.projecteuler.net. 

Solution checking is performed via pytest unit tests. 

## Tricks/Methods

#### Problem 5

Given a set of necessary divisors, find the smallest number with that set among its divisors. The prime factorization of each divisor will be a subset of the prime factorisation of the number. Find the prime factorisations of the divisors, then find the max power of each prime necessary to create each divisor. These powers are the used to find the final answer
$$
n = \prod_i^kp_i^{e_i}
$$

#### Problem 8

Use a moving window across the list of digits, find the max.

#### Problem 9

use euclid formula to get integer solutions to pythagoras eq, under the constraint of a+b+c=p. 

#### Problem 11

iterate through matrix elements with double for loop, create horizontal, vertical, right diagonal and left diagonal groups of 4 for each index, then add to a product list. return the max valued product.

#### Problem 12

Use a prime factorization method (trial division up to square root). Then use fact that number of divisors satisfies
$$
d(n) = \prod_{i=1}^k (e_i+1)
$$
where e_k are the powers of the prime factor representation of n. 

#### Problem 15

This is a graph path counting problem. Create the adjacency matrix of the graph (directed graph so not symmetric, no loops). The number of paths of length k between vertex i and j is given by the power of the adjacency matrix.
$$
C_k = A^k
$$


#### Problem 16

dynamic typing makes this easy, just get 2**1000 then convert to string then to individual digits and sum

#### Problem 18

Each of the paths through the triangle can be represented by a binary number. There are 2^N total paths where N is the depth of the triangle. AS you loop through the enumerated paths, find the binary number, then the column index for that row in the path is the sum of all the binary digits up to that row. Loop through all paths to find max value.

#### Problem 19

Create function to get the n days in a month given a year and month. iterate through the days of the month and years, counting the days, alongside a day of the week counter. if day of week counter evenly divisible by 7 then it is a sunday, then check if it is the first of the month.

#### Problem 20

calculate factorial, convert to string, then sum digits

#### Problem 21

Use prime factorization then sum of divisors using
$$
\sigma(n) = \prod_{i=1}^k \frac{p_i^{e_i + 1} -1 }{p_i -1}
$$
find all the amicable numbers by simple search of all below max number.

#### Problem 22

load in file with `with open` then trip and split the names into a list. Sort with `sorted`, then use `enumerate` to get index and multiply with name score. This names score comes from `dict` with enumerated alphabet values.

#### Problem 23

Find all abundant numbers below the max value of 28124, then find all possible pair sums. Using set difference, find those numbers below 28124 which do not exist in this combination set. return the sum of these.

#### Problem 24

Create the lexicographic permutations of the numbers 0 to 9. There is a standard djikstra's algorithm for this. Find the rightmost index that is larger than its left neighbour, then swap all the elements up to the next value. Then 

#### Problem 25

Find first fibonacci number with specific number of digits.

Use dummy variables to count up the fibonacci numbers in a while loop up until the `len(str(fn))` reaches right length.

#### Problem 26

Perform division, where we use `//` and `%` to find the quotient and remainder respectively. For each integer, perform division in a while loop to find all the digits of the result, breaking if the remainder hits zero or if a remainder is repeated. There will be a reciprocal cycle if a remainder is repeated, the length of the reciprocal cycle is the number of digits before a repeated remainder. 

#### Problem 27

for consecutive prime results to quadratic 
$$
n^2 + an+b
$$
Iterate through a nested for loop of a and b. Then nest a while loop that checks each consecutive result is prime (using the prime factorization). Return that with the longest sequence of consecutive prime results.

#### Problem 28

Number Spiral: iterate through the spiral diagonals by  using a for loop and updating the interval every time you reach a layer max. All diagonal values are odd , and the intervals increase each layer by 2.

#### Problem 30

Simple search and check with an if statement. Create a list of powers of digits, then for each number below some chosen large value, get the digits, then check the sum of the powers of said digits. Include to list.

#### Problem 31

Could possibly do this recursively, but instead, do nested for loop for decrease coin size, where the max N of each coin size for loop is set by the number of above sizes. Check to see if sum equals 200, and count.

#### Problem 32

Use the lexicographic permutation method to create all permutations of the digits 1-9. Then search all "slices" of these permutations e.g. "234589761" ->  "234", "5", "89761" by taking slices of the string and check if each satisfies a*b=c.

#### Problem 33

loop through double-digit denominators and numerators up to denominator. Find a common digit that is not 0. Check if the reduced fraction is equal to the original fraction. 

#### Problem 34

iterate through numbers, use `str` and `math` package `factorial` to check if sum of factorials of digits equals number. Just chose a large number to stop at (100000).

#### Problem 35

Generate list of primes with eratosthenes, then rotate the primes by shifting the string version with array slicing. Check each rotation is prime by getting its prime factors. Add to circular prime list if all rotations are prime.

#### Problem 36

Use the bin function to find binary version of number, then use reversed to check if palindrome; add to list and sum.

#### Problem 37

Use eratothenes sieve to create a list of primes up to a arbitrary large number. I know there are 11 to look for so just choose the number until 11 matching primes are reached. Check for truncation by iterating over the left and right sequences of string-ified primes and check if each sub-integer is a prime.

#### Problem 38

Use lexicographic permutation algorithm to loop through all permutations of digits. Check for pandigital multiple by popping first n digits  off digits list and checking if the next few digits satisfy 1*y, 2* y, 3*y in a while loop. If the whole digits list is popped empty, then it is a pandigital multiple. 

#### Problem 39

Like problem 9 , find pythagorean triplet solutions using euclid formula. remember to remove duplicate solutions, then search for the best in range 1<=p<=1000

#### Problem 40

create champerdowne constant by string concatenation, then just find the product of digits and index locations.

#### Problem 41

for a set of lists of digits from 3 to n, loop through the lexicographic permutations and check if the resultant integer is a prime by checking the number of prime factors.

#### Problem 42

Calculate word values then check how many are in a list of triangle numbers.

#### Problem 43

Again loop through the lexicographical permutations and search for those with the required properties. 

#### Problem 44

Likely a much more elegant solution, but I just searched through the pairs of pentagonal numbers, (lower triangular) and checked if the sum and difference were also pentagonal.

#### Problem 45

Again simply loop through the triangle numbers up to some specified max, checking each for pentagonal and hexagonal numberness.

#### Problem 46

Use the eratosthenes sieve function to find all primes below each odd composite number, then calculate each value of the squared number to see if it is an integer (using floor(x)==x). If there are no integer values of this variable for every prime below c then it is the solution.  

#### Problem 47

create a list of consecutive numbers, find their prime factors then check the length of the set{}, if all prime factor sets the right length, return the result

#### Problem 48

Use binary exponentiation to find large self powers, then sum up 

#### Problem 52

while loop iterating up through the positive numbers, look at a sorted list of digits, and check if all the sorted digit lists of all multiples are equal

#### Problem 53

Combinatorial problem. Computing the binomial coefficients every time would be very expensive so just precompute the factorials for all integers up to n before the loop (using a python dict). then just search the binomial coefficients for those more than a million.

#### Problem 56

Use binary exponentiation to find large powers, convert to strings then sum digits. Find the largest

#### Problem 57

Use recurrence relations for the convergents of an infinite continued fraction.
$$
h_n = a_n h_{n-1} + h_{n-2}\\
k_n = a_n k_{n-1} + h_{n-2}
$$
For the numerator and denominator of the nth rational approximation respectively.

#### Problem 63

Use fact that the number of digits in decimal number can be found with log
$$
N_d = \lfloor\log_{10}(X)\rfloor +1
$$
Problem is then to find the number of solutions (d, x) that satisfy
$$
d-1 = \lfloor d\log_{10}(x)\rfloor
$$
For a specific value of d, that largest value x can take is 10, and so loop over values of d and x = 1-10 and count those that satisfy the above expression.

#### Problem 64

We get the continued fraction for the square root of a non-square number $N$ by
$$
r = \sqrt{N}\\
i = \lfloor r\rfloor \\
f = r - i \\
r\prime = 1/f
$$
the continued fraction representation will be 
$$
a_n = i_n
$$
We use the fact that the final $a_n$ of a periodic sequence in the continued fraction representation is twice the leading term.
$$
a_n = 2a_0
$$

#### Problem 69

Use the euler totient function $$\phi(n)$$ (calculated from the prime factors of n) to find the ratio of  n/totient. Then simply use a running best to find the value of n for which the ratio is maximized.

#### Problem 92

For each integer, while loop to iterate through the number chain, using `str` to get the digits of each integer. Break the loop if you find and 89 or 1. Then just counts the occurrences of 89 as you go

#### Problem 97

Python can handle large numbers and so the trivial solution of just calculating the large prime and finding the last 10 digits is possible by brute force. However a more refined solution is to 

use modulo 
$$
D_n(x) = \lfloor (x/10^{n-1}) \mod 10  \rfloor
$$
