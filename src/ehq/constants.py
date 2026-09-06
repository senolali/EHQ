"""Framework-wide constants with no runtime side effects."""

FRAMEWORK_VERSION = "0.4.0"
PROTOCOL_VERSION = "1.1.4"

# Provider request/response identity is independent of EHQ post-processing.
# The response cache is keyed on what was actually sent to and returned by a
# provider, so it is pinned to the last framework version whose request
# construction changed. Bumping FRAMEWORK_VERSION for a scoring-protocol change
# must not silently invalidate retained provider responses.
RESPONSE_IDENTITY_VERSION = "0.2.12"

# The registry is the record of every route considered, not the list of routes
# that scored. Its size is fixed so a truncated or duplicated file cannot be
# loaded unnoticed; it is raised deliberately, with the added entry carrying its
# own dated verification evidence, when a route is registered.
REFERENCE_PANEL_SIZE = 21

RESPONSE_LABELS = (
    "ABSTAIN",
    "HEDGE",
    "CONFIDENT_CORRECT",
    "CONFIDENT_WRONG",
)

# Official EHQ3 definition: calibration is measured only where the model chose
# to give a substantive answer. Restraint behaviour (ABSTAIN/HEDGE) is already
# measured by EHQ1 and EHQ2 and must not be double-counted here.
EHQ3_PROTOCOL = "confidence_substantive_only_v1"
EHQ3_SUBSTANTIVE_LABELS = ("CONFIDENT_CORRECT", "CONFIDENT_WRONG")

TECHNICAL_FAILURE_LABELS = (
    "API_ERROR",
    "INVALID_RESPONSE",
    "INVALID_CONFIDENCE",
)

DATASET_CATEGORIES = ("FEQ", "PCQ", "HNQ", "CCQ")
REDACTION_TOKEN = "[REDACTED]"
