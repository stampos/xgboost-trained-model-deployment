import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class PlaygroundUITests(unittest.TestCase):
    def open_app(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def assert_healthy(self, app):
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    def test_training_and_both_prediction_modes_with_intermediate_inputs(self):
        app = self.open_app()
        self.assertTrue(app.button(key="predict").disabled)
        self.assertNotIn("Μέλι100", app.selectbox(key="liquid").options)
        app.button(key="train").click().run()
        self.assert_healthy(app)
        self.assertIn("144 εγγραφές εκπαίδευσης / 36 ελέγχου", app.success[0].value)
        app.number_input(key="height").set_value(17.5)
        app.number_input(key="mass_Κύβος").set_value(39.5)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        direct = app.metric[-1].value
        self.assertEqual(app.metric[-1].label, "Εκτιμώμενος χρόνος")
        app.radio(key="prediction_mode").set_value("Με παρεμβολή").run()
        # Old predictions disappear as soon as an input changes.
        self.assertEqual(len(app.metric), 3)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        self.assertNotEqual(direct, app.metric[-1].value)
        self.assertEqual(len(app.dataframe[-1].value), 4)

    def test_model_and_dataset_changes_block_stale_predictions(self):
        app = self.open_app()
        app.button(key="train").click().run()
        app.selectbox(key="model").set_value("Polynomial Ridge").run()
        self.assertTrue(app.button(key="predict").disabled)
        self.assertFalse(app.success)
        app.button(key="train").click().run()
        self.assert_healthy(app)
        self.assertIn("Polynomial Ridge", app.success[0].value)
        app.selectbox(key="dataset").set_value("Με Μέλι100").run()
        self.assertTrue(app.button(key="predict").disabled)
        self.assertIn("Μέλι100", app.selectbox(key="liquid").options)
        app.button(key="train").click().run()
        self.assert_healthy(app)
        self.assertIn("168 εγγραφές εκπαίδευσης / 42 ελέγχου", app.success[0].value)

    def test_full_dataset_missing_combination_and_return_to_reduced_dataset(self):
        app = self.open_app()
        app.selectbox(key="dataset").set_value("Με Μέλι100").run()
        app.button(key="train").click().run()
        app.selectbox(key="liquid").set_value("Μέλι100").run()
        self.assertTrue(app.button(key="predict").disabled)
        self.assertTrue(app.warning)
        app.number_input(key="mass_Κύβος").set_value(77.0)
        app.number_input(key="height").set_value(17.5)
        app.radio(key="prediction_mode").set_value("Με παρεμβολή").run()
        self.assertFalse(app.button(key="predict").disabled)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        self.assertEqual(len(app.dataframe[-1].value), 2)
        app.number_input(key="mass_Κύβος").set_value(39.5).run()
        self.assertTrue(app.button(key="predict").disabled)
        app.radio(key="prediction_mode").set_value("Χωρίς παρεμβολή").run()
        self.assertFalse(app.button(key="predict").disabled)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        app.selectbox(key="shape").set_value("Σφαίρα").run()
        app.number_input(key="mass_Σφαίρα").set_value(35.5)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        app.selectbox(key="dataset").set_value("Χωρίς Μέλι100").run()
        self.assert_healthy(app)
        self.assertNotIn("Μέλι100", app.selectbox(key="liquid").options)
        self.assertNotEqual(app.selectbox(key="liquid").value, "Μέλι100")
        self.assertTrue(app.button(key="predict").disabled)


if __name__ == "__main__":
    unittest.main()
