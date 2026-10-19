from kb.metadata import basic_metadata, detect_chapters, keyword_labels


def test_title_and_author_from_title_page():
    meta = basic_metadata("THE TIME MACHINE\nAn Invention\nby H. G. Wells\n", {}, "scan.pdf")
    assert meta["title"] == "The Time Machine: An Invention"
    assert meta["author"] == "H. G. Wells"


def test_embedded_pdf_metadata_wins():
    meta = basic_metadata("Some first line\n", {"title": "Real Title", "author": "Real Author"}, "x.pdf")
    assert (meta["title"], meta["author"]) == ("Real Title", "Real Author")


def test_gutenberg_header():
    text = "Title: The Time Machine\nAuthor: H. G. Wells\nLanguage: English\n"
    meta = basic_metadata(text, {}, "pg35.txt")
    assert meta["author"] == "H. G. Wells"
    assert meta["language"] == "English"


def test_falls_back_to_filename():
    assert basic_metadata("", {}, "annual_report-2025.pdf")["title"] == "Annual Report 2025"


def test_detect_chapters():
    pages = ["Chapter I: Introduction\n\nText.", "More text.", "Chapter II: The Machine\n\nText."]
    assert detect_chapters(pages) == [
        {"heading": "Chapter I: Introduction", "page": 1},
        {"heading": "Chapter II: The Machine", "page": 3},
    ]


def test_keyword_labels_skip_stopwords():
    labels = keyword_labels("The machine and the machine and the time traveller travelled through time time")
    assert labels[0] == "time"
    assert "the" not in labels
