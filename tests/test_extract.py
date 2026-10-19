from kb.extract import join_lines


def test_lines_with_normal_spacing_stay_in_one_paragraph():
    lines = [(0, 12, "I would like to express my gratitude to Dr. Coumar,"),
             (14, 26, "Department of Bioinformatics, for his support.")]
    assert "\n\n" not in join_lines(lines)


def test_large_gap_starts_new_paragraph():
    lines = [(0, 12, "First paragraph ends here."), (14, 26, "Still first."),
             (45, 57, "Second paragraph.")]
    assert join_lines(lines) == "First paragraph ends here.\nStill first.\n\nSecond paragraph."


def test_double_spaced_document():
    # Measured from a real double-spaced thesis: line gap 14.3, paragraph gap 23.8, line height 13.3
    tops, lines = 0.0, []
    for text, gap in [("Machine learning has become", 0), ("integral to genomics.", 14.3),
                      ("It handles", 14.3), ("big data.", 14.3), ("2. Autophagy and Cancer", 23.8),
                      ("Studies have identified", 7.1), ("related genes.", 14.4)]:
        tops += gap
        lines.append((tops, tops + 13.3, text))
        tops += 13.3
    paras = join_lines(lines).split("\n\n")
    assert len(paras) == 2
    assert paras[1].startswith("2. Autophagy and Cancer")


def test_jump_back_up_the_page_is_a_new_column():
    lines = [(500, 512, "End of left column."), (100, 112, "Top of right column.")]
    assert join_lines(lines) == "End of left column.\n\nTop of right column."


def test_empty():
    assert join_lines([]) == ""
