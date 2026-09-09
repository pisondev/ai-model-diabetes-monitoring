"""Chart colours. Light only, because the app itself is pinned to light in .streamlit/config.toml."""

SURFACE = "#ffffff"
INK = "#0f172a"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# categorical slots, taken in this fixed order and never cycled
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
ACCENT = SERIES[0]
EMPHASIS = SERIES[1]
DEEMPHASIS = "#cbd5e1"

NEGATIVE_LABEL = "not diabetic"
POSITIVE_LABEL = "diabetic"
CLASS_RANGE = (SERIES[0], SERIES[1])

# one hue, light to dark, gaps wide enough to read as discrete steps
ORDINAL = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")

# warm and cool poles with a neutral midpoint, for correlation where the sign is the point
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")
