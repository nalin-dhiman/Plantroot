from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_starts_without_exception() -> None:
    app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    assert not test.exception
    assert len(test.tabs) == 4

    rerun = test.run()
    assert not rerun.exception
    assert len(rerun.tabs) == 4


def test_application_module_starts_without_exception() -> None:
    app = Path(__file__).resolve().parents[1] / "app" / "main.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    assert not test.exception
