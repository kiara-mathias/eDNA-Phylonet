"""Theme CSS is loadable and targets the Streamlit widgets we restyle."""

from __future__ import annotations

from theme.inject import load_css


def test_load_css_overrides_core_streamlit_widgets():
    css = load_css()
    assert "--edna-sand:" in css
    assert "[data-testid=\"stTextInput\"]" in css
    assert "[data-testid=\"stButton\"]" in css or ".stButton" in css
    assert "[data-testid=\"stAlert\"]" in css
    assert "[data-testid=\"stDataFrame\"]" in css
    assert "[data-testid=\"stTabs\"]" in css
    assert "IBM Plex Mono" in css
    assert "Inter" in css
    assert ".family-icon" in css
    assert ".result-callout" in css
    assert ".fcw-callout" in css
    assert "background-attachment: fixed" in css
    assert ".record-header::before" in css
    assert ".lie-card" in css
