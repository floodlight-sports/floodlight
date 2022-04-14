from floodlight.core.pitch import Pitch

#: Sample football pitch of size 105m x 68m with coordinate origin in the center.
pitch = Pitch(
    xlim=(-52.5, 52.5),
    ylim=(-34, 34),
    unit="m",
    boundaries="flexible",
    length=105,
    width=68,
    sport="football",
)
