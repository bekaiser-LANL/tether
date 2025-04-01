"""Tests of module1 """

import unittest
from my_library.module1 import add

class TestModule1(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)

if __name__ == '__main__':
    unittest.main()
