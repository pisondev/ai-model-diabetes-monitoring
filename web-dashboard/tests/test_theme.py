import tomllib
from pathlib import Path

from lib.ui import light_mode_css

WEB_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = WEB_ROOT / ".streamlit" / "config.toml"


def theme():
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)["theme"]


def luminance(colour):
    red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255


def test_theme_is_pinned_so_the_system_scheme_cannot_win():
    assert theme()["base"] == "light"


def test_every_themed_surface_is_light_and_its_text_is_dark():
    palette = theme()
    assert luminance(palette["backgroundColor"]) > 0.9
    assert luminance(palette["secondaryBackgroundColor"]) > 0.85
    assert luminance(palette["textColor"]) < 0.3


def test_css_declares_the_light_colour_scheme_to_the_browser():
    assert "color-scheme: light" in light_mode_css()


def test_css_takes_its_colours_from_the_config_rather_than_its_own_copy():
    assert theme()["secondaryBackgroundColor"] in light_mode_css()


def test_every_screen_goes_through_the_helper_that_injects_the_css():
    screens = [WEB_ROOT / "Home.py", *sorted((WEB_ROOT / "pages").glob("*.py"))]
    assert len(screens) == 5
    for screen in screens:
        source = screen.read_text(encoding="utf-8")
        assert "from lib.ui import" in source
        assert "page(" in source
