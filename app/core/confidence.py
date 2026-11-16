"""The most important file in the project.

Every prediction needs a confidence readout that sounds extremely scientific.
Accuracy is optional. Confidence is not.
"""


def format_confidence(p: float) -> dict:
    pct = round(p * 100, 1)
    if pct >= 90:
        note = "the oracle has never been more certain"
    elif pct >= 70:
        note = "the oracle is fairly sure and slightly smug"
    else:
        note = "the oracle is guessing but will not admit it"
    return {
        "pct": f"{pct}%",
        "note": note,
        "margin": "±0.0% (we did not check)",
    }
