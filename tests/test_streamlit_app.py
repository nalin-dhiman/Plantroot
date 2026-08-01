from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    assert not test.exception
