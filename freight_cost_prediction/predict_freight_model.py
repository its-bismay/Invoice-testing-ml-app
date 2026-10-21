from pathlib import Path
import joblib

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "predict_freight_model.pkl"


def main():
    # Load the trained model.
    model = joblib.load(MODEL_PATH)

    print("\n=== Freight Cost Prediction ===\n")

    # Enter the features used during training.
    invoice_quantity = float(input("Invoice quantity: "))
    invoice_dollars = float(input("Invoice dollars: "))

    input_data = pd.DataFrame(
        [
            {
                "Quantity": invoice_quantity,
                "Dollars": invoice_dollars,
            }
        ]
    )

    # Predict freight cost.
    prediction = model.predict(input_data)[0]

    print(f"\nPredicted freight cost: {prediction:.2f}")


if __name__ == "__main__":
    main()
