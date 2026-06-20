def test_fuzzywuzzy_importable_and_ratio_works():
    from fuzzywuzzy import fuzz

    assert fuzz.ratio("hello world", "hello world") == 100
    assert fuzz.ratio("abc", "xyz") < 50
