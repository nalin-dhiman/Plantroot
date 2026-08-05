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


def test_extended_preview_runs_with_shared_worker_guardrails() -> None:
    app = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    test.radio[0].set_value("Extended preview · up to 30 d").run()
    test.select_slider[0].set_value(7)
    next(button for button in test.button if button.label == "Grow this root").click().run()

    assert not test.exception
    assert any("Extended preview" in warning.value for warning in test.warning)
    guarded = {
        button.label: button.disabled
        for button in test.button
        if button.label in {"Run paired comparison", "Run resolution check"}
    }
    assert guarded == {"Run paired comparison": True, "Run resolution check": True}
