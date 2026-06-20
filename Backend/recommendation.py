"""
recommendation.py — maps a severity tier to a concrete operational
recommendation (manpower, barricading, alerting).

Kept as a separate module from inference.py on purpose: the ML model's job
is "how severe is this likely to be", the recommendation layer's job is
"given that severity, what should we actually DO". Mixing these into one
function would make it harder to explain or adjust the action mapping
without touching model code.

This is intentionally a simple, transparent lookup -- not another model.
For a hackathon prototype, a clear and defensible rule-based mapping is
more credible to judges than a black-box "recommendation model" you can't
fully explain on the spot.
"""

TIER_ACTIONS = {
    "HIGH": {
        "action": "deploy_full_response",
        "manpower_units": 2,
        "barricade": True,
        "alert_control_room": True,
        "message_template": (
            "Sustained disruption likely. Deploy {manpower_units} patrol units "
            "and set up barricade on {corridor}."
        ),
    },
    "MEDIUM": {
        "action": "flag_for_review",
        "manpower_units": 1,
        "barricade": False,
        "alert_control_room": True,
        "message_template": (
            "Uncertain severity on {corridor}. Flagging for dispatcher review; "
            "1 unit on standby, no barricade yet."
        ),
    },
    "ROUTINE": {
        "action": "log_only",
        "manpower_units": 0,
        "barricade": False,
        "alert_control_room": False,
        "message_template": (
            "Likely routine event on {corridor}. Logged, no active deployment."
        ),
    },
}


def get_recommendation(tier: str, corridor: str) -> dict:
    """
    tier     : 'HIGH' | 'MEDIUM' | 'ROUTINE' (from inference.apply_tier)
    corridor : the corridor name from the input event, used to personalize
               the human-readable message.
    """
    if tier not in TIER_ACTIONS:
        raise ValueError(f"Unknown tier: {tier!r}. Expected HIGH, MEDIUM, or ROUTINE.")

    spec = TIER_ACTIONS[tier]
    message = spec["message_template"].format(
        manpower_units=spec["manpower_units"],
        corridor=corridor or "this corridor",
    )

    return {
        "action": spec["action"],
        "manpower_units": spec["manpower_units"],
        "barricade": spec["barricade"],
        "alert_control_room": spec["alert_control_room"],
        "message": message,
    }