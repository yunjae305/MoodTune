import unittest

from streamlit.testing.v1 import AppTest


class VectorPageTests(unittest.TestCase):
    def test_vector_db_page_renders_filter_controls(self):
        at = AppTest.from_file("app.py")
        at.session_state["view"] = "vector_db"

        at.run()

        self.assertGreaterEqual(len(at.selectbox), 5)
        self.assertGreaterEqual(len(at.slider), 2)
        self.assertGreaterEqual(len(at.text_input), 3)

    def test_benchmark_page_renders_query_and_run_controls(self):
        at = AppTest.from_file("app.py")
        at.session_state["view"] = "benchmark"

        at.run()

        self.assertGreaterEqual(len(at.slider), 1)
        self.assertEqual(len(at.text_input), 1)


if __name__ == "__main__":
    unittest.main()
