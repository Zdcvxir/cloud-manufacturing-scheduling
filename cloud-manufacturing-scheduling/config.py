"""Benchmark instance used by the scheduling algorithms.

The lowercase variable names mirror the notation used in the paper. Clear
uppercase aliases are provided for readers who are less familiar with the
mathematical symbols.
"""

# Problem size
N = NUM_JOBS = 200
T = NUM_SLOTS = 50
B = MAX_BATCHES = 200

# Model parameters
a = POSITION_EXPONENT = 0.1
seta = SETUP_TIME = 2
c = BATCH_CAPACITY = 3
w = FAILURE_COST_WEIGHT = 5
u = PM_SETUP_COST = 30
k = FAILURE_RATE_SCALE = 0.05
lam = FAILURE_RATE = 0.01

# Slot-specific processing cost coefficient.
tao_l = SLOT_PROCESSING_COSTS = [
    0.876638532423534, 1.1127453568345995, 0.8828367833648813,
    1.0739437301599355, 0.9748536349868304, 0.8661601416993009,
    0.8170643314719603, 1.0837359853687139, 0.8434589808407384,
    1.192735974996067, 0.94374598867309, 0.9699410889060016,
    0.9204908264933217, 0.9278828702941911, 0.8954591176882616,
    0.8817893038084849, 0.9975360825141275, 0.9406852631123654,
    0.8266996950010368, 1.0334594133148873, 1.1316524826874204,
    0.9285102857239161, 0.8300963377045772, 0.9847678378689518,
    1.160639256062154, 1.0462356849661076, 1.1420085637990827,
    1.0917540594129371, 0.9246936253847374, 0.9689981878346565,
    0.8586507918065566, 0.8211161109603369, 0.8696941677499813,
    0.8255485504754221, 0.89412601081245, 1.0028640080906024,
    0.8480388825998897, 1.170552678775629, 0.965933825655781,
    0.9822701024565702, 1.129451956343261, 1.1847082607562465,
    0.9536318171671002, 1.1948015769841067, 0.8962639436253709,
    0.8052255491015976, 1.047032850391876, 0.8876046862673747,
    1.1109458897076276, 0.9102094591784479,
]

# Job processing times.
p_j = JOB_PROCESSING_TIMES = [
    4, 4, 4, 4, 5, 4, 6, 4, 4, 6, 6, 4, 5, 4, 4, 5, 6, 4, 5, 4,
    4, 5, 4, 5, 5, 4, 4, 6, 6, 5, 5, 4, 6, 5, 5, 6, 5, 4, 6, 4,
    6, 6, 6, 5, 4, 5, 6, 5, 4, 5, 6, 6, 6, 5, 5, 4, 4, 6, 6, 6,
    5, 4, 5, 6, 6, 6, 5, 4, 6, 5, 5, 4, 6, 6, 4, 4, 4, 6, 5, 4,
    6, 5, 4, 5, 6, 4, 4, 4, 5, 6, 4, 5, 4, 6, 6, 5, 4, 6, 4, 6,
    6, 4, 4, 5, 4, 6, 6, 4, 4, 4, 6, 6, 4, 5, 6, 5, 4, 5, 6, 4,
    6, 4, 6, 5, 6, 6, 6, 6, 5, 6, 6, 6, 4, 4, 4, 5, 6, 6, 6, 6,
    6, 6, 6, 6, 4, 6, 4, 4, 5, 5, 6, 6, 6, 4, 4, 5, 5, 5, 4, 5,
    5, 4, 6, 5, 5, 5, 6, 6, 5, 5, 4, 6, 6, 6, 5, 5, 6, 4, 4, 6,
    5, 5, 4, 6, 5, 6, 6, 4, 4, 5, 5, 6, 6, 4, 5, 6, 5, 6, 6, 6,
]

# Maximum available length of each slot.
L_l = SLOT_LENGTHS = [
    79, 80, 76, 78, 79, 63, 61, 62, 70, 67, 77, 73, 60, 60, 63, 76,
    69, 62, 61, 78, 76, 69, 77, 71, 70, 65, 71, 75, 80, 61, 68, 79,
    78, 63, 69, 77, 71, 73, 72, 63, 77, 65, 80, 71, 66, 70, 63, 70,
    63, 73,
]

# Fixed cost of selecting each slot.
CP_l = SLOT_FIXED_COSTS = [
    388, 242, 265, 457, 265, 495, 469, 340, 425, 272, 335, 235, 254,
    409, 469, 428, 293, 485, 479, 359, 432, 328, 294, 332, 432, 422,
    214, 364, 376, 251, 260, 439, 341, 496, 462, 283, 491, 293, 232,
    490, 363, 251, 289, 305, 345, 231, 326, 233, 272, 285,
]


def validate_instance() -> None:
    """Validate basic dimensions before an algorithm starts."""
    if len(p_j) != N:
        raise ValueError(f"Expected {N} job processing times, got {len(p_j)}.")
    for name, values in {
        "tao_l": tao_l,
        "L_l": L_l,
        "CP_l": CP_l,
    }.items():
        if len(values) != T:
            raise ValueError(f"Expected {T} values in {name}, got {len(values)}.")
    if c <= 0:
        raise ValueError("Batch capacity c must be positive.")
    if B <= 0:
        raise ValueError("Maximum number of batches B must be positive.")


validate_instance()
