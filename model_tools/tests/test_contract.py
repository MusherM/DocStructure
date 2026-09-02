from model_tools.contract import GEARS, gear_for_image


def test_visual_token_gears_are_expected() -> None:
    assert [gear.visual_tokens for gear in GEARS] == [192, 432, 768, 1320]


def test_selects_smallest_area_gear() -> None:
    assert gear_for_image(100, 100).name == "384x512"
    assert gear_for_image(700, 900).name == "768x1024"
    assert gear_for_image(2000, 2000).name == "960x1408"
