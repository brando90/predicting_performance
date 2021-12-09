import unittest

import automl.utils.utils_datasets as utils


class TestStringMethods(unittest.TestCase):

    def test_imports_through_packages(self):
        helloworld = utils.HelloWorld()
        self.assertTrue( helloworld == 'HelloWorld')


if __name__ == '__main__':
    unittest.main()
