from pathlib import Path
import pickle
import joblib

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

INVOICE_MODEL_PATH = BASE_DIR / "invoice_flagging" / "model" / "invoice_risk_model.pkl"

FREIGHT_MODEL_PATH = (
    BASE_DIR / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"
)


# -------------------- Page Configuration --------------------

st.set_page_config(
    page_title="Invoice ML Pro",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Invoice ML Pro")
st.caption(
    "A simple machine learning dashboard for invoice risk detection "
    "and freight cost estimation."
)

st.divider()


# -------------------- Load Models --------------------


@st.cache_resource
def load_invoice_model():
    with open(INVOICE_MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_freight_model():
    return joblib.load(FREIGHT_MODEL_PATH)


# -------------------- Navigation --------------------

risk_tab, freight_tab = st.tabs(["🛡️ Invoice Risk", "🚚 Freight Cost"])


# ==================== INVOICE RISK ====================

with risk_tab:
    st.subheader("Invoice Risk Detection")
    st.write(
        "Enter invoice and purchase details to check whether "
        "an invoice should be reviewed."
    )

    with st.container(border=True):
        st.markdown("#### Invoice Details")

        col1, col2 = st.columns(2)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value=0,
                value=100,
                step=1,
            )

            invoice_dollars = st.number_input(
                "Invoice Dollars ($)",
                min_value=0.0,
                value=18500.0,
                step=100.0,
            )

            freight = st.number_input(
                "Freight ($)",
                min_value=0.0,
                value=250.0,
                step=10.0,
            )

        with col2:
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=0,
                value=100,
                step=1,
            )

            total_item_dollars = st.number_input(
                "Total Item Dollars ($)",
                min_value=0.0,
                value=18500.0,
                step=100.0,
            )

        predict_risk = st.button(
            "Check Invoice Risk",
            type="primary",
            use_container_width=True,
        )

    if predict_risk:
        try:
            model = load_invoice_model()

            input_data = pd.DataFrame(
                [
                    {
                        "invoice_quantity": invoice_quantity,
                        "invoice_dollars": invoice_dollars,
                        "Freight": freight,
                        "total_item_quantity": total_item_quantity,
                        "total_item_dollars": total_item_dollars,
                    }
                ]
            )

            prediction = model.predict(input_data)[0]

            st.markdown("### Prediction Result")

            if prediction == 1:
                st.error(
                    "Invoice flagged for manual review",
                    icon="⚠️",
                )
                st.write("The model classified this invoice as potentially risky.")
            else:
                st.success(
                    "Invoice classified as normal",
                    icon="✅",
                )
                st.write("The model did not flag this invoice for review.")

        except Exception as error:
            st.error(f"Could not predict invoice risk: {error}")


# ==================== FREIGHT COST ====================

with freight_tab:
    st.subheader("Freight Cost Estimation")
    st.write(
        "Enter the quantity and dollar value of the items "
        "to estimate their freight cost."
    )

    with st.container(border=True):
        st.markdown("#### Purchase Details")

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value=0,
                value=100,
                step=1,
                key="freight_quantity",
                help="Enter the number of items.",
            )

        with col2:
            dollars = st.number_input(
                "Dollars ($)",
                min_value=0.0,
                value=18500.0,
                step=100.0,
                key="freight_dollars",
                help="Enter the total dollar value of the items.",
            )

        st.markdown("#### Input Summary")

        summary1, summary2 = st.columns(2)
        summary1.metric("Total Quantity", f"{quantity:,}")
        summary2.metric("Item Value", f"${dollars:,.2f}")

        predict_freight = st.button(
            "Estimate Freight Cost",
            type="primary",
            use_container_width=True,
        )

    if predict_freight:
        try:
            model = load_freight_model()

            # Match the feature names used during model training.
            input_data = pd.DataFrame(
                {
                    "Quantity": [quantity],
                    "Dollars": [dollars],
                }
            )

            prediction = model.predict(input_data)[0]

            st.markdown("### Prediction Result")

            st.metric(
                label="Estimated Freight Cost",
                value=f"${prediction:,.2f}",
            )

            st.caption("This is a model estimate, not a guaranteed shipping quote.")

        except Exception as error:
            st.error(f"Could not predict freight cost: {error}")


# -------------------- Footer --------------------

st.divider()

st.caption("Invoice ML Pro · Machine learning predictions for testing and analysis")
