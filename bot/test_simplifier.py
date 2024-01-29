import unittest
from simplifier import Simplifier


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.__simplifier = Simplifier()

    def test_empty_sent(self):
        empty = ''
        result_sent = self.__simplifier.simplify_sent(empty)
        self.assertIsInstance(result_sent, str)
        self.assertEqual(result_sent, '')

    def test_empty_full(self):
        empty = ''
        result_full = self.__simplifier.simplify(empty)
        self.assertIsInstance(result_full, str)
        self.assertEqual(result_full, '')

    def test_sent(self):
        sent = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский, после чего поселковый совет стал называться Советским.'
        result_sent = self.__simplifier.simplify_sent(sent)
        self.assertIsInstance(result_sent, str)
        self.assertGreater(len(result_sent), 1)

    def test_full(self):
        text = '14 декабря 1944 года рабочий посёлок Ички был переименован в рабочий посёлок Советский. После этого поселковый совет стал называться Советским.'
        result_full = self.__simplifier.simplify_sent(text)
        self.assertIsInstance(result_full, str)
        self.assertGreater(len(result_full), 1)


if __name__ == '__main__':
    unittest.main()