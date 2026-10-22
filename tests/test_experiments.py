from mot_track.experiments import BYTE_GRID, SORT_GRID, byte_params, fmt_row, grid, header


def test_grid_is_full_product():
    g = grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6
    assert {"a": 2, "b": 5} in g
    assert len(grid(SORT_GRID)) == 54 and len(grid(BYTE_GRID)) == 64


def test_byte_params_sets_new_track_threshold():
    assert byte_params({"track_thresh": 0.6})["new_track_thresh"] == 0.7


def test_markdown_row_matches_header():
    m = {k: 1.0 for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")}
    m.update({k: 3 for k in ("IDSW", "FP", "FN", "MT", "ML", "Frag")})
    head = header(["run"]).splitlines()[0]
    assert head.count("|") == fmt_row("x", m).count("|")
