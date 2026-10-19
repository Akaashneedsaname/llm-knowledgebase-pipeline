from kb.chunk import chunk_pages, count_tokens


def _para(n_words: int, word: str = "lorem") -> str:
    return " ".join([word] * n_words)


def test_respects_token_budget_and_keeps_order():
    pages = ["\n\n".join(_para(40, f"w{i}") for i in range(10))]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Each chunk may exceed the budget only by the overlap seed
    assert all(c.token_count <= 100 + 10 for c in chunks)


def test_overlap_carries_text_into_next_chunk():
    pages = ["\n\n".join(_para(60, f"w{i}") for i in range(4))]
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=15)
    tail_word = chunks[0].text.split()[-1]
    assert chunks[1].text.startswith(tail_word) or tail_word in chunks[1].text.split()[:20]


def test_tracks_pages_and_chapters():
    pages = [
        "Chapter I: Introduction\n\n" + _para(30, "alpha"),
        _para(30, "beta"),
        "Chapter II: The Machine\n\n" + _para(30, "gamma"),
    ]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=0)
    assert [c.chapter for c in chunks] == ["Chapter I: Introduction", "Chapter II: The Machine"]
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert (chunks[1].page_start, chunks[1].page_end) == (3, 3)


def test_rejoins_sentence_split_across_pages():
    pages = ["The gay", "robes of the beautiful people moved."]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=0)
    assert chunks[0].text == "The gay robes of the beautiful people moved."
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_tiny_preamble_merges_into_first_chapter():
    pages = ["THE TIME MACHINE\n\nChapter XIII: The Trap\n\n" + _para(30, "alpha")]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].chapter == "Chapter XIII: The Trap"


def test_splits_oversized_paragraph():
    chunks = chunk_pages([_para(500)], chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 5
    assert all(count_tokens(c.text) <= 110 for c in chunks)


def test_empty_document():
    assert chunk_pages(["", "   "]) == []
