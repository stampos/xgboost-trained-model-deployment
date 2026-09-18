import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Πρόβλεψη Χρόνου Κίνησης",
    page_icon="🔬",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("xgboost_model.joblib")


model = load_model()


# ============================================================
# CONSTANTS
# ============================================================

KNOWN_HEIGHTS = np.array(
    [5, 10, 15, 20, 25, 30],
    dtype=float
)


OBJECTS = {
    "Κύβος 2 g": ("Κύβος", 2),
    "Κύβος 77 g": ("Κύβος", 77),
    "Σφαίρα 13 g": ("Σφαίρα", 13),
    "Σφαίρα 59 g": ("Σφαίρα", 59),
    "Δίσκος 11 g": ("Δίσκος", 11),
    "Δίσκος 66 g": ("Δίσκος", 66)
}


LIQUIDS = [
    "Νερό",
    "Αλατόνερο",
    "Λάδι",
    "Μέλι30",
    "Μέλι70",
    "Μέλι100"
]


# ============================================================
# INVALID COMBINATIONS
# ============================================================

INVALID_COMBINATIONS = {
    (
        "Μέλι100",
        "Κύβος",
        2
    ):
        "Ο Κύβος 2 g δεν έπεσε στο Μέλι100 και δεν υπάρχουν "
        "πειραματικές μετρήσεις για αυτόν τον συνδυασμό."
}


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_with_interpolation(
    model,
    liquid,
    height,
    shape,
    mass
):
    """
    Για γνωστό πειραματικό ύψος χρησιμοποιεί απευθείας
    την πρόβλεψη του XGBoost.

    Για ενδιάμεσο ύψος χρησιμοποιεί τις XGBoost προβλέψεις
    στα δύο γειτονικά γνωστά ύψη και εφαρμόζει
    γραμμική παρεμβολή.

    Για μη διαθέσιμους πειραματικούς συνδυασμούς
    δεν πραγματοποιείται πρόβλεψη.
    """

    height = float(height)

    # ========================================================
    # ΕΛΕΓΧΟΣ ΜΗ ΕΓΚΥΡΩΝ ΣΥΝΔΥΑΣΜΩΝ
    # ========================================================

    combination = (
        liquid,
        shape,
        mass
    )

    if combination in INVALID_COMBINATIONS:
        raise ValueError(
            INVALID_COMBINATIONS[combination]
        )

    # ========================================================
    # ΕΛΕΓΧΟΣ ΟΡΙΩΝ ΥΨΟΥΣ
    # ========================================================

    if (
        height < KNOWN_HEIGHTS.min()
        or height > KNOWN_HEIGHTS.max()
    ):
        raise ValueError(
            "Το ύψος πρέπει να βρίσκεται μεταξύ 5 και 30 cm."
        )

    # ========================================================
    # ΠΕΡΙΠΤΩΣΗ 1:
    # ΓΝΩΣΤΟ ΠΕΙΡΑΜΑΤΙΚΟ ΥΨΟΣ
    # ========================================================

    if np.any(
        np.isclose(
            KNOWN_HEIGHTS,
            height
        )
    ):

        input_data = pd.DataFrame({
            "Υγρό": [liquid],
            "Ύψος (cm)": [height],
            "Σχήμα": [shape],
            "Μάζα (g)": [mass]
        })

        prediction = model.predict(
            input_data
        )[0]

        return float(prediction), {
            "method": "direct"
        }

    # ========================================================
    # ΠΕΡΙΠΤΩΣΗ 2:
    # ΕΝΔΙΑΜΕΣΟ ΥΨΟΣ -> INTERPOLATION
    # ========================================================

    upper_index = np.searchsorted(
        KNOWN_HEIGHTS,
        height
    )

    lower_height = (
        KNOWN_HEIGHTS[
            upper_index - 1
        ]
    )

    upper_height = (
        KNOWN_HEIGHTS[
            upper_index
        ]
    )

    # --------------------------------------------------------
    # XGBoost predictions στα δύο γειτονικά ύψη
    # --------------------------------------------------------

    anchor_data = pd.DataFrame({
        "Υγρό": [
            liquid,
            liquid
        ],

        "Ύψος (cm)": [
            lower_height,
            upper_height
        ],

        "Σχήμα": [
            shape,
            shape
        ],

        "Μάζα (g)": [
            mass,
            mass
        ]
    })


    predictions = model.predict(
        anchor_data
    )


    lower_prediction = float(
        predictions[0]
    )

    upper_prediction = float(
        predictions[1]
    )


    # ========================================================
    # LINEAR INTERPOLATION
    # ========================================================

    weight = (
        (height - lower_height)
        /
        (upper_height - lower_height)
    )


    prediction = (
        lower_prediction
        +
        weight
        *
        (
            upper_prediction
            -
            lower_prediction
        )
    )


    interpolation_info = {

        "method": "interpolation",

        "lower_height":
            float(lower_height),

        "upper_height":
            float(upper_height),

        "lower_prediction":
            lower_prediction,

        "upper_prediction":
            upper_prediction,

        "weight":
            float(weight)
    }


    return (
        float(prediction),
        interpolation_info
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔬 Πρόβλεψη Χρόνου Κίνησης"
)


st.write(
    """
    Η εφαρμογή χρησιμοποιεί ένα εκπαιδευμένο μοντέλο
    **XGBoost** για την πρόβλεψη του χρόνου κίνησης
    ενός αντικειμένου μέσα σε διαφορετικά υγρά.

    Για ύψη που βρίσκονται μεταξύ δύο πειραματικών
    μετρήσεων εφαρμόζεται **γραμμική παρεμβολή**
    μεταξύ των αντίστοιχων προβλέψεων του XGBoost.
    """
)


# ============================================================
# INPUTS
# ============================================================

st.subheader(
    "Εισαγωγή χαρακτηριστικών"
)


# ------------------------------------------------------------
# Liquid
# ------------------------------------------------------------

liquid = st.selectbox(
    "Υγρό",
    LIQUIDS
)


# ------------------------------------------------------------
# Height
# ------------------------------------------------------------

height = st.number_input(
    "Ύψος (cm)",
    min_value=5.0,
    max_value=30.0,
    value=15.0,
    step=0.5,
    format="%.1f"
)


# ------------------------------------------------------------
# Object
# ------------------------------------------------------------

object_choice = st.selectbox(
    "Αντικείμενο",
    list(
        OBJECTS.keys()
    )
)


shape, mass = (
    OBJECTS[
        object_choice
    ]
)


# ============================================================
# CHECK CURRENT COMBINATION
# ============================================================

current_combination = (
    liquid,
    shape,
    mass
)


combination_is_valid = (
    current_combination
    not in INVALID_COMBINATIONS
)


if not combination_is_valid:

    st.warning(
        "⚠️ Ο συγκεκριμένος συνδυασμός δεν είναι διαθέσιμος."
    )

    st.error(
        INVALID_COMBINATIONS[
            current_combination
        ]
    )

else:

    st.caption(
        "Μπορούν να χρησιμοποιηθούν και ενδιάμεσες "
        "τιμές ύψους μεταξύ 5 και 30 cm."
    )


# ============================================================
# SHOW CURRENT INPUT
# ============================================================

st.divider()


st.write(
    "### Επιλεγμένα χαρακτηριστικά"
)


col1, col2 = st.columns(2)


with col1:

    st.write(
        f"**Υγρό:** {liquid}"
    )

    st.write(
        f"**Ύψος:** {height:.1f} cm"
    )


with col2:

    st.write(
        f"**Σχήμα:** {shape}"
    )

    st.write(
        f"**Μάζα:** {mass} g"
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()


# Το button απενεργοποιείται αν ο συνδυασμός είναι μη έγκυρος
prediction_button = st.button(
    "Υπολογισμός πρόβλεψης",
    type="primary",
    use_container_width=True,
    disabled=not combination_is_valid
)


if prediction_button:

    try:

        prediction, info = (
            predict_with_interpolation(
                model=model,
                liquid=liquid,
                height=height,
                shape=shape,
                mass=mass
            )
        )


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.success(
            f"Προβλεπόμενος χρόνος: "
            f"{prediction:.5f} s"
        )


        st.metric(
            label="Εκτιμώμενος χρόνος",
            value=f"{prediction:.5f} s"
        )


        # ====================================================
        # DIRECT XGBOOST
        # ====================================================

        if info["method"] == "direct":

            st.info(
                f"Το ύψος {height:.1f} cm "
                "αντιστοιχεί σε πειραματικό ύψος "
                "του dataset. Χρησιμοποιήθηκε "
                "απευθείας η πρόβλεψη του XGBoost."
            )


        # ====================================================
        # INTERPOLATION
        # ====================================================

        elif (
            info["method"]
            ==
            "interpolation"
        ):

            lower_height = (
                info[
                    "lower_height"
                ]
            )

            upper_height = (
                info[
                    "upper_height"
                ]
            )

            lower_prediction = (
                info[
                    "lower_prediction"
                ]
            )

            upper_prediction = (
                info[
                    "upper_prediction"
                ]
            )


            st.info(
                f"Το ύψος {height:.1f} cm "
                "δεν είναι ένα από τα αρχικά "
                "πειραματικά ύψη. "
                "Η τελική τιμή υπολογίστηκε "
                "με γραμμική παρεμβολή μεταξύ "
                f"{lower_height:g} cm και "
                f"{upper_height:g} cm."
            )


            st.write(
                "### Υπολογισμός παρεμβολής"
            )


            interp_col1, interp_col2 = (
                st.columns(2)
            )


            with interp_col1:

                st.metric(
                    label=(
                        "XGBoost στα "
                        f"{lower_height:g} cm"
                    ),

                    value=(
                        f"{lower_prediction:.5f} s"
                    )
                )


            with interp_col2:

                st.metric(
                    label=(
                        "XGBoost στα "
                        f"{upper_height:g} cm"
                    ),

                    value=(
                        f"{upper_prediction:.5f} s"
                    )
                )


            st.write(
                f"""
                Η πρόβλεψη για **{height:.1f} cm**
                βρίσκεται μεταξύ των δύο παραπάνω
                προβλέψεων.
                """
            )


    # ========================================================
    # INVALID COMBINATION
    # ========================================================

    except ValueError as e:

        st.error(
            f"⚠️ Μη διαθέσιμη πρόβλεψη\n\n{e}"
        )


    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as e:

        st.error(
            "Παρουσιάστηκε απρόβλεπτο σφάλμα "
            f"κατά την πρόβλεψη: {e}"
        )


# ============================================================
# METHODOLOGY INFORMATION
# ============================================================

st.divider()


with st.expander(
    "ℹ️ Πώς υπολογίζεται η πρόβλεψη;"
):

    st.write(
        """
        Τα πειραματικά ύψη που χρησιμοποιήθηκαν
        για την εκπαίδευση είναι:

        **5, 10, 15, 20, 25 και 30 cm**

        Αν ο χρήστης επιλέξει ένα από αυτά τα ύψη,
        χρησιμοποιείται απευθείας η πρόβλεψη
        του XGBoost.

        Αν επιλεγεί ενδιάμεσο ύψος, όπως για
        παράδειγμα **17.5 cm**, η εφαρμογή:

        1. Υπολογίζει την πρόβλεψη του XGBoost
        στο αμέσως μικρότερο πειραματικό ύψος.

        2. Υπολογίζει την πρόβλεψη του XGBoost
        στο αμέσως μεγαλύτερο πειραματικό ύψος.

        3. Εφαρμόζει γραμμική παρεμβολή μεταξύ
        των δύο τιμών.

        Η εφαρμογή πραγματοποιεί προβλέψεις μόνο
        εντός του πειραματικού εύρους **5–30 cm**.

        Επίσης, δεν πραγματοποιείται πρόβλεψη για
        συνδυασμούς για τους οποίους δεν υπάρχουν
        πειραματικές μετρήσεις και δεν παρατηρήθηκε
        κίνηση του αντικειμένου.
        """
    )


# ============================================================
# DATASET LIMITATIONS
# ============================================================

with st.expander(
    "⚠️ Περιορισμοί δεδομένων"
):

    st.write(
        """
        Για τον συνδυασμό:

        **Μέλι100 + Κύβος 2 g**

        δεν υπάρχουν μετρήσεις χρόνου στο dataset,
        καθώς ο συγκεκριμένος κύβος δεν έπεσε στο υγρό.

        Για αυτόν τον λόγο η εφαρμογή δεν επιτρέπει
        την παραγωγή πρόβλεψης για τη συγκεκριμένη
        περίπτωση.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Εφαρμογή πρόβλεψης βασισμένη σε XGBoost "
    "και γραμμική παρεμβολή για ενδιάμεσα ύψη."
)