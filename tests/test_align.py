from dealpoint.data.align import align_span, fragments


def test_fragments_splits_on_omitted_and_blank_line():
    raw = (
        "First excerpt here that is long enough.<omitted>Second excerpt also long enough"
        "\n\nThird excerpt also long enough."
    )
    frags = fragments(raw)
    assert len(frags) == 3


def test_fragments_strips_page_markers():
    raw = "This is a long enough excerpt (Page 12) with a page marker embedded in it."
    frags = fragments(raw)
    assert len(frags) == 1
    assert "(Page" not in frags[0]
    assert "Page 12" not in frags[0]


def test_fragments_drops_short_parts():
    raw = "ok<omitted>x"
    frags = fragments(raw)
    assert frags == []


def test_align_span_exact_match():
    canonical = "prefix text " + "A" * 50 + " suffix text"
    raw_span = "A" * 50
    result = align_span(canonical, raw_span)
    assert result.n_fragments == 1
    assert result.n_aligned == 1
    assert result.modes == ["exact"]
    assert result.covered_fraction == 1.0
    assert result.ranges == [(12, 62)]


def test_align_span_fuzzy_match_on_single_char_corruption():
    base = "The quick brown fox jumps over the lazy dog and keeps running fast"
    canonical = "prefix " + base + " suffix"
    corrupted = base[:30] + "X" + base[31:]  # single character swapped
    result = align_span(canonical, corrupted)
    assert result.n_fragments == 1
    assert result.n_aligned == 1
    assert result.modes == ["fuzzy"]


def test_align_span_with_omitted_marker():
    canonical = "AAAAAAAAAAAAAAAAAAAA middle text here BBBBBBBBBBBBBBBBBBBB"
    raw_span = "AAAAAAAAAAAAAAAAAAAA<omitted>BBBBBBBBBBBBBBBBBBBB"
    result = align_span(canonical, raw_span)
    assert result.n_fragments == 2
    assert result.n_aligned == 2
    assert result.ranges == [(0, 20), (38, 58)]


def test_align_span_with_page_marker_in_raw_span():
    long_text = "This clause discusses material adverse effects in significant detail here"
    canonical = "intro " + long_text + " outro"
    raw_span = long_text[:40] + " (Page 5) " + long_text[40:]
    result = align_span(canonical, raw_span)
    assert result.n_aligned >= 1


def test_align_span_unalignable_returns_empty_not_raise():
    canonical = "short document with nothing matching"
    raw_span = "completely unrelated text that appears nowhere in the document at all"
    result = align_span(canonical, raw_span)
    assert result.ranges == []
    assert result.n_aligned == 0
    assert result.covered_fraction == 0.0
