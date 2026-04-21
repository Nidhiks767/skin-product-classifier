from main_pipeline import clean_text, rule_based

def test_clean_text():
    assert clean_text("SPF 50!!!") == "spf 50"

def test_rule_based():
    assert rule_based("spf sunscreen lotion") == "Sunscreen"
    assert rule_based("niacinamide serum") == "Serum"
    assert rule_based("face wash gel") == "Cleanser"