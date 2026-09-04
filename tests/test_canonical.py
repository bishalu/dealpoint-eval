from dealpoint.data.canonical import canonicalise, normalise_quote


def test_idempotence():
    raw = "Hello\r\n\r\nWorld\u2019s   \u201cbest\u201d\xa0deal\u2013maker\u2014ever\u200b"
    once = canonicalise(raw)
    twice = canonicalise(once)
    assert once == twice


def test_bom_stripped():
    assert canonicalise("\ufeffHello").startswith("Hello")


def test_nfkc_normalisation():
    # Full-width digit normalises to ASCII digit under NFKC.
    assert canonicalise("\uff11\uff12\uff13") == "123"


def test_curly_quotes_to_ascii():
    assert canonicalise("\u2018a\u2019 \u201cb\u201d") == "'a' \"b\""


def test_dashes_to_hyphen():
    assert canonicalise("a\u2013b\u2014c\u2212d") == "a-b-c-d"


def test_nbsp_to_space():
    assert canonicalise("a\xa0b") == "a b"


def test_zero_width_space_removed():
    assert canonicalise("a\u200bb") == "ab"


def test_crlf_and_cr_to_newline_then_collapsed():
    assert canonicalise("a\r\nb\rc") == "a b c"


def test_whitespace_collapsed_including_newlines():
    assert canonicalise("a   \n\n  b\tc") == "a b c"


def test_strip_leading_trailing_whitespace():
    assert canonicalise("   hello   ") == "hello"


def test_golden_fixture():
    raw = "\ufeffExhibit 2.1 \r\n\r\n\r\nAGREEMENT   AND\tPLAN \u2018Test\u2019 \u201cQuote\u201d\r\n"
    assert canonicalise(raw) == 'Exhibit 2.1 AGREEMENT AND PLAN \'Test\' "Quote"'


def test_normalise_quote_is_alias_of_canonicalise():
    text = "  some\r\nmodel   output\u2019s quote  "
    assert normalise_quote(text) == canonicalise(text)
