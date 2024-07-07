import enum


class Stage(enum.StrEnum):
    """Enum for the stage of the training"""

    TRAIN = "train"
    VALID = "val"
    TEST = "test"
