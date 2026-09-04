import pytest

from dealpoint.data.canonical import canonicalise, normalise_quote

pytestmark = pytest.mark.gate_m1


def test_verbatim_normalisation_matches_after_curly_quotes_and_dashes():
    canonical_text = canonicalise(
        "The parties agree that the \u2018Purchase Price\u2019 shall be \u201cUSD 100\u2014150 million\u201d "
        "as adjusted, with a non\u2011breaking\u00a0space in between."
    )
    # A model quote using curly quotes / en-dash / nbsp for the same passage.
    model_quote = (
        "the \u2018Purchase Price\u2019 shall be \u201cUSD 100\u2014150 million\u201d as adjusted, with a "
        "non\u2011breaking\u00a0space in between"
    )
    normalised_quote = normalise_quote(model_quote)
    assert normalised_quote in canonical_text


def test_verbatim_normalisation_collapses_whitespace_differences():
    canonical_text = canonicalise("Section 6.3 states that the Company shall indemnify Parent.")
    model_quote = "the Company   shall\nindemnify Parent"
    assert normalise_quote(model_quote) in canonical_text


def test_fabricated_quote_does_not_match():
    canonical_text = canonicalise("Section 6.3 states that the Company shall indemnify Parent.")
    fabricated = "the Company shall indemnify Buyer for all losses whatsoever"
    assert normalise_quote(fabricated) not in canonical_text
