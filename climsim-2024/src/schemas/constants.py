import enum


class Phase(enum.StrEnum):
    """Phase of the training loop."""

    TRAINING = "Training"
    VALIDATION = "Validation"
