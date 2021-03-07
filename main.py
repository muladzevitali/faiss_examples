import sys
import unittest


def test():
    tests = unittest.TestLoader().discover('tests', pattern='test_*.py')
    unittest.TextTestRunner(verbosity=1).run(tests)


if __name__ == '__main__':
    if sys.argv[1] == 'test':
        test()
        sys.exit(0)
