from kb.clean import clean_text, paragraphs, remove_repeated_lines


def test_rejoins_hyphenated_words():
    assert clean_text("the experi-\nment worked") == "the experiment worked"


def test_unwraps_lines_but_keeps_paragraphs():
    text = "First line of a\nparagraph.\n\nSecond\nparagraph."
    assert paragraphs(clean_text(text)) == ["First line of a paragraph.", "Second paragraph."]


def test_removes_bare_page_numbers():
    assert clean_text("Some text.\n\n42\n\nMore text.") == "Some text.\n\nMore text."


def test_keeps_chapter_heading_as_own_paragraph():
    text = "Chapter IV: Time Travelling\nI told some of you last Thursday"
    assert paragraphs(clean_text(text)) == ["Chapter IV: Time Travelling", "I told some of you last Thursday"]


def test_removes_running_headers_and_footers():
    bodies = ["The fire burnt brightly.", "Filby became pensive.", "The lamp flickered.",
              "Nobody believed him.", "It was a strange tale."]
    pages = [f"THE TIME MACHINE\n{body}\nPage {i}" for i, body in enumerate(bodies, start=1)]
    cleaned = remove_repeated_lines(pages)
    assert all("THE TIME MACHINE" not in p and "Page" not in p for p in cleaned)
    assert cleaned[2] == "The lamp flickered."


def test_keeps_similar_body_lines_in_middle_of_page():
    pages = [f"HEADER\nIntro.\nTable {i}: yields by year\nOutro.\nFOOTER" for i in range(1, 6)]
    assert all(f"Table {i}: yields by year" in p for i, p in enumerate(remove_repeated_lines(pages), start=1))


def test_never_removes_chapter_headings():
    pages = [f"Chapter {i}\nStory text {i}." for i in range(1, 6)]
    assert remove_repeated_lines(pages)[0].startswith("Chapter 1")
