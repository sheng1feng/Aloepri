from src.security_qwen import default_gate6_cases


def test_default_gate6_cases_have_two_targeted_variants() -> None:
    names = [item.name for item in default_gate6_cases()]
    assert names == ["targeted_mild", "targeted_strong", "targeted_extreme"]
