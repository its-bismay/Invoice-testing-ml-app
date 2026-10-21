from pathlib import Path
import pickle

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "invoice_risk_model.pkl"


def main():
    # Load the trained model.
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("\n=== Invoice Risk Prediction ===\n")

    # Enter invoice details.
    invoice_quantity = float(input("Invoice quantity: "))
    invoice_dollars = float(input("Invoice dollars: "))
    freight = float(input("Freight: "))
    total_item_quantity = float(input("Total item quantity: "))
    total_item_dollars = float(input("Total item dollars: "))

    # Prepare input in the same feature order used during training.
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

    # Predict invoice risk.
    prediction = model.predict(input_data)[0]

    print("\n=== Prediction Result ===")

    if prediction == 1:
        print("Invoice flagged for manual review.")
    else:
        print("Invoice classified as normal.")


if __name__ == "__main__":
    main()
