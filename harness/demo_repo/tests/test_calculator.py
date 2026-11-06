"""Tests for the demo repo. test_add FAILS on the buggy code; the other
two pass and form the capped regression suite."""
import unittest

from calculator import add, multiply, subtract


class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)


if __name__ == "__main__":
    unittest.main()
