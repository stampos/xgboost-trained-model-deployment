import re
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class LanguageUITests(unittest.TestCase):
    def assert_healthy(self, app):
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    def test_english_controls_tables_and_dynamic_validation(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.selectbox(key="language").set_value("en").run()
        self.assert_healthy(app)
        self.assertEqual(app.title[0].value, "🔬 Motion Time Prediction")
        self.assertEqual(app.selectbox(key="dataset").options, ["Without Honey100", "With Honey100"])
        self.assertEqual(app.selectbox(key="liquid").options, ["Water", "Salt water", "Oil", "Honey30", "Honey70"])
        self.assertEqual(app.selectbox(key="shape").options, ["Cube", "Sphere", "Disk"])
        self.assertEqual(app.selectbox(key="model").label, "Model")
        self.assertEqual(app.number_input(key="height").label, "Height (cm)")
        self.assertEqual(app.number_input(key="mass_Κύβος").label, "Mass (g)")
        self.assertEqual(app.radio(key="prediction_mode").options, ["Without interpolation", "With interpolation"])
        self.assertEqual(app.button(key="train").label, "Train model")
        self.assertEqual(app.button(key="predict").label, "Predict")
        table = app.dataframe[0].value
        self.assertEqual(list(table.columns), ["Liquid", "Height (cm)", "Shape", "Mass (g)", "Mean time (s)"])
        self.assertFalse(re.search(r"[\u0370-\u03ff]", table.to_csv(index=False)))
        app.selectbox(key="dataset").set_value("Με Μέλι100").run()
        app.selectbox(key="liquid").set_value("Μέλι100").run()
        self.assert_healthy(app)
        self.assertIn("2 g Cube in Honey100", app.warning[0].value)
        self.assertTrue(app.button(key="predict").disabled)
        app.number_input(key="mass_Κύβος").set_value(39.5)
        app.radio(key="prediction_mode").set_value("Με παρεμβολή").run()
        self.assert_healthy(app)
        self.assertIn("mass (g)", app.warning[0].value)
        self.assertIn("77", app.warning[0].value)
        self.assertFalse(re.search(r"[\u0370-\u03ff]", app.warning[0].value))
        self.assertTrue(app.button(key="predict").disabled)

    def test_language_switch_preserves_training_inputs_and_predictions(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.selectbox(key="dataset").set_value("Με Μέλι100")
        app.selectbox(key="model").set_value("Random Forest").run()
        app.selectbox(key="liquid").set_value("Μέλι70")
        app.selectbox(key="shape").set_value("Σφαίρα").run()
        app.number_input(key="height").set_value(17.5)
        app.number_input(key="mass_Σφαίρα").set_value(35.5)
        app.radio(key="prediction_mode").set_value("Με παρεμβολή")
        app.button(key="train").click().run()
        experiment = app.session_state["experiment"]
        app.button(key="predict").click().run()
        greek_prediction = app.metric[-1].value

        app.selectbox(key="language").set_value("en").run()
        self.assert_healthy(app)
        self.assertIs(app.session_state["experiment"], experiment)
        expected = {"dataset": "Με Μέλι100", "model": "Random Forest", "liquid": "Μέλι70", "shape": "Σφαίρα"}
        for key, value in expected.items():
            self.assertEqual(app.selectbox(key=key).value, value)
        self.assertEqual(app.number_input(key="height").value, 17.5)
        self.assertEqual(app.number_input(key="mass_Σφαίρα").value, 35.5)
        self.assertEqual(app.radio(key="prediction_mode").value, "Με παρεμβολή")
        self.assertFalse(app.button(key="predict").disabled)
        self.assertIn("168 training records / 42 test records", app.success[0].value)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        self.assertEqual(app.metric[-1].label, "Estimated time")
        self.assertEqual(app.metric[-1].value, greek_prediction)
        self.assertIn("Interpolation weight", app.dataframe[-1].value.columns)
        self.assertFalse(re.search(r"[\u0370-\u03ff]", app.dataframe[-1].value.to_csv(index=False)))

        app.radio(key="prediction_mode").set_value("Χωρίς παρεμβολή")
        app.button(key="predict").click().run()
        english_direct_prediction = app.metric[-1].value
        app.selectbox(key="language").set_value("el").run()
        self.assert_healthy(app)
        self.assertIs(app.session_state["experiment"], experiment)
        self.assertEqual(app.selectbox(key="shape").value, "Σφαίρα")
        self.assertEqual(app.number_input(key="mass_Σφαίρα").value, 35.5)
        self.assertFalse(app.button(key="predict").disabled)
        app.button(key="predict").click().run()
        self.assert_healthy(app)
        self.assertEqual(app.metric[-1].label, "Εκτιμώμενος χρόνος")
        self.assertEqual(app.metric[-1].value, english_direct_prediction)


if __name__ == "__main__":
    unittest.main()
