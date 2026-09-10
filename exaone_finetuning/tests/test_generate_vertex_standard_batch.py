import unittest
from unittest.mock import patch

from exaone_finetuning.generate_vertex_standard_batch import (
    generate_with_backoff,
    is_resource_exhausted,
)


class ResourceExhausted(Exception):
    status_code = 429


class VertexStandardBatchTests(unittest.TestCase):
    def test_detects_resource_exhausted(self):
        self.assertTrue(is_resource_exhausted(ResourceExhausted("capacity")))
        self.assertTrue(is_resource_exhausted(Exception("429 RESOURCE_EXHAUSTED")))
        self.assertFalse(is_resource_exhausted(Exception("bad request")))

    @patch("exaone_finetuning.generate_vertex_standard_batch.time.sleep")
    def test_retries_429_and_returns_result(self, sleep):
        results = iter([ResourceExhausted("capacity"), "done"])

        def call():
            value = next(results)
            if isinstance(value, Exception):
                raise value
            return value

        self.assertEqual(generate_with_backoff(call, attempts=2, initial_wait=3), "done")
        sleep.assert_called_once_with(3)

    def test_does_not_retry_other_errors(self):
        def call():
            raise ValueError("invalid")

        with self.assertRaises(ValueError):
            generate_with_backoff(call)


if __name__ == "__main__":
    unittest.main()
