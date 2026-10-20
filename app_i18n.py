"""Display translations; model inputs always retain their original CSV values."""


MESSAGES = {
    "el": {
        "title": "Πρόβλεψη Χρόνου Κίνησης",
        "intro": "Επίλεξε δεδομένα και μοντέλο, εκπαίδευσέ το και δοκίμασε διαφορετικά ύψη και μάζες, με ή χωρίς παρεμβολή.",
        "training_heading": "1. Δεδομένα και εκπαίδευση",
        "model": "Μοντέλο",
        "read_error": "Δεν ήταν δυνατή η ανάγνωση του dataset: {error}",
        "dataset_summary": "{rows} εγγραφές · {liquids} υγρά. Η επιλογή dataset αφορά το Μέλι100· τα Μέλι30 και Μέλι70 περιλαμβάνονται και στα δύο.",
        "split_help": "Κάθε μοντέλο εκπαιδεύεται στο ίδιο 80% των εγγραφών. Το υπόλοιπο 20% κρατείται για έλεγχο και δεν χρησιμοποιείται στην εκπαίδευση.",
        "train": "Εκπαίδευση μοντέλου",
        "training": "Εκπαίδευση {model}…",
        "training_error": "Η εκπαίδευση δεν ολοκληρώθηκε: {error}",
        "trained": "Εκπαιδευμένο μοντέλο: {model} · {dataset} · {train} εγγραφές εκπαίδευσης / {test} ελέγχου.",
        "fit_time": "Χρόνος εκπαίδευσης: {seconds:.2f} s. Μετρικές στο 20% ελέγχου:",
        "metrics_help": "Οι μετρικές αφορούν τις απευθείας προβλέψεις του μοντέλου, χωρίς παρεμβολή.",
        "stale": "Άλλαξες dataset ή μοντέλο. Πάτησε «Εκπαίδευση μοντέλου» για τη νέα επιλογή.",
        "not_ready": "Πάτησε «Εκπαίδευση μοντέλου» για να ενεργοποιήσεις τις προβλέψεις.",
        "view_data": "Προβολή δεδομένων",
        "prediction_heading": "2. Χαρακτηριστικά και πρόβλεψη",
        "measured": "Μετρημένα ύψη: {heights} cm · Μετρημένες μάζες: {masses} g.",
        "intermediate_help": "Μπορείς να πληκτρολογήσεις και ενδιάμεσες τιμές ύψους και μάζας.",
        "prediction_mode": "Τρόπος πρόβλεψης",
        "interpolation_help": "Γραμμική παρεμβολή στο ύψος και στη μάζα, ανάμεσα στις προβλέψεις του μοντέλου στα γειτονικά μετρημένα σημεία.",
        "direct_help": "Το μοντέλο προβλέπει απευθείας για το ύψος και τη μάζα που επέλεξες.",
        "honey_cube_help": "Στο Μέλι100 υπάρχουν μετρήσεις κύβου μόνο στα 77 g. Η παρεμβολή υποστηρίζεται μόνο σε αυτή τη μάζα.",
        "predict": "Υπολογισμός πρόβλεψης",
        "estimated_time": "Εκτιμώμενος χρόνος",
        "negative_time": "Το μοντέλο επέστρεψε αρνητικό χρόνο, που δεν έχει φυσική ερμηνεία.",
        "interpolated_result": "Η πρόβλεψη προέκυψε με παρεμβολή στις προβλέψεις του μοντέλου.",
        "anchors": "Σημεία και συντελεστές παρεμβολής",
        "exact_result": "Το σημείο είναι μετρημένος συνδυασμός· χρησιμοποιήθηκε απευθείας το μοντέλο.",
        "direct_result": "Χρησιμοποιήθηκε η απευθείας πρόβλεψη του μοντέλου, χωρίς παρεμβολή.",
        "prediction_error": "Δεν ήταν δυνατή η πρόβλεψη: {error}",
        "interpretation": "Πώς να ερμηνεύσεις τις δοκιμές",
        "interpretation_help": "Οι ενδιάμεσες μάζες και τα ενδιάμεσα ύψη είναι εκτιμήσεις, όχι νέες πειραματικές μετρήσεις. Η παρεμβολή χρησιμοποιεί τις προβλέψεις του επιλεγμένου μοντέλου, όχι τους καταγεγραμμένους χρόνους. Οι επιλογές περιορίζονται στο εύρος ύψους και μάζας που υπάρχει για κάθε σχήμα.",
        "exploratory_help": "Οι δοκιμές μοντέλων εδώ είναι διερευνητικές. Για επιλογή μοντέλου με διασταυρούμενη επικύρωση χρησιμοποίησε την ξεχωριστή ανάλυση του έργου.",
        "session": "Η εκπαίδευση ισχύει για την τρέχουσα συνεδρία της εφαρμογής.",
        "invalid_model": "Επίλεξε ένα από τα διαθέσιμα μοντέλα.",
        "interpolation_bounds": "Η παρεμβολή για {axis} απαιτεί τιμή από {low:g} έως {high:g} για το συγκεκριμένο υγρό και σχήμα.",
        "mass_axis": "τη μάζα (g)",
        "height_axis": "το ύψος (cm)",
        "nonfinite_inputs": "Το ύψος και η μάζα πρέπει να είναι πεπερασμένοι αριθμοί.",
        "unknown_combination": "Δεν υπάρχουν δεδομένα για αυτό το υγρό και σχήμα.",
        "height_bounds": "Το ύψος πρέπει να βρίσκεται εντός του πειραματικού εύρους.",
        "mass_bounds": "Η μάζα πρέπει να βρίσκεται εντός του εύρους για το σχήμα.",
        "missing_honey_cube": "Δεν υπάρχουν μετρήσεις για Κύβο 2 g στο Μέλι100.",
        "no_shared_heights": "Δεν υπάρχουν κοινά μετρημένα ύψη για την παρεμβολή.",
        "invalid_time": "Το μοντέλο δεν επέστρεψε έγκυρο χρόνο.",
    },
    "en": {
        "title": "Motion Time Prediction",
        "intro": "Choose a dataset and model, train it, then try different heights and masses, with or without interpolation.",
        "training_heading": "1. Data and training",
        "model": "Model",
        "read_error": "Could not read the dataset: {error}",
        "dataset_summary": "{rows} records · {liquids} liquids. The dataset choice includes or excludes Honey100; Honey30 and Honey70 are available in both.",
        "split_help": "Every model trains on the same 80% of records. The remaining 20% is held out for testing and is never used for training.",
        "train": "Train model",
        "training": "Training {model}…",
        "training_error": "Training could not be completed: {error}",
        "trained": "Trained model: {model} · {dataset} · {train} training records / {test} test records.",
        "fit_time": "Training time: {seconds:.2f} s. Metrics on the held-out 20% test set:",
        "metrics_help": "These metrics evaluate direct model predictions, without interpolation.",
        "stale": "You changed the dataset or model. Click “Train model” for the new selection.",
        "not_ready": "Click “Train model” to enable predictions.",
        "view_data": "View data",
        "prediction_heading": "2. Inputs and prediction",
        "measured": "Measured heights: {heights} cm · Measured masses: {masses} g.",
        "intermediate_help": "You can also enter intermediate heights and masses.",
        "prediction_mode": "Prediction method",
        "interpolation_help": "Linear interpolation in height and mass between the model predictions at neighboring measured coordinates.",
        "direct_help": "The model predicts directly for your selected height and mass.",
        "honey_cube_help": "Cube measurements in Honey100 are available only at 77 g. Interpolation is supported only at that mass.",
        "predict": "Predict",
        "estimated_time": "Estimated time",
        "negative_time": "The model returned a negative time, which has no physical interpretation.",
        "interpolated_result": "The result was interpolated between model predictions.",
        "anchors": "Interpolation points and weights",
        "exact_result": "This is a measured combination; the model was used directly.",
        "direct_result": "The model prediction was used directly, without interpolation.",
        "prediction_error": "Could not make a prediction: {error}",
        "interpretation": "How to interpret your experiments",
        "interpretation_help": "Intermediate masses and heights are estimates, not new experimental measurements. Interpolation uses predictions from the selected model, not recorded times. Inputs stay within the measured height and mass range for each shape.",
        "exploratory_help": "These model experiments are exploratory. For model selection with cross-validation, use the project's separate analysis.",
        "session": "Training is kept for the current app session.",
        "invalid_model": "Choose one of the available models.",
        "interpolation_bounds": "Interpolation for {axis} requires a value from {low:g} to {high:g} for this liquid and shape.",
        "mass_axis": "mass (g)",
        "height_axis": "height (cm)",
        "nonfinite_inputs": "Height and mass must be finite numbers.",
        "unknown_combination": "No data are available for this liquid and shape.",
        "height_bounds": "Height must be within the measured range.",
        "mass_bounds": "Mass must be within the measured range for this shape.",
        "missing_honey_cube": "No measurements are available for a 2 g Cube in Honey100.",
        "no_shared_heights": "No shared measured heights are available for interpolation.",
        "invalid_time": "The model did not return a valid time.",
    },
}

ENGLISH_LABELS = {
    "Χωρίς Μέλι100": "Without Honey100", "Με Μέλι100": "With Honey100",
    "Νερό": "Water", "Αλατόνερο": "Salt water", "Λάδι": "Oil",
    "Μέλι30": "Honey30", "Μέλι70": "Honey70", "Μέλι100": "Honey100",
    "Κύβος": "Cube", "Σφαίρα": "Sphere", "Δίσκος": "Disk",
    "Υγρό": "Liquid", "Ύψος (cm)": "Height (cm)", "Σχήμα": "Shape",
    "Μάζα (g)": "Mass (g)", "Μέσος χρόνος (s)": "Mean time (s)",
    "Χωρίς παρεμβολή": "Without interpolation", "Με παρεμβολή": "With interpolation",
    "Πρόβλεψη μοντέλου (s)": "Model prediction (s)",
    "Συντελεστής παρεμβολής": "Interpolation weight",
}


def translate(language, key, **values):
    return MESSAGES[language][key].format(**values)


def display_label(value, language):
    return ENGLISH_LABELS.get(value, value) if language == "en" else value


def display_table(df, language):
    """Translate a copy, leaving training features and targets untouched."""
    result = df.copy()
    if language == "en":
        for column in ("Υγρό", "Σχήμα"):
            if column in result:
                result[column] = result[column].map(lambda value: display_label(value, language))
        result = result.rename(columns=ENGLISH_LABELS)
    return result
