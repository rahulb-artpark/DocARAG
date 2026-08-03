"""A composite figure must not reach the tree as one node per panel.

Page 5 of the Celestron EdgeHD paper is a single figure: a 3x5 matrix of spot
diagrams comparing three telescope designs across five off-axis distances. It
arrived as fifteen image blocks. Each was cropped, described and indexed alone,
so the document's record of it was fifteen variations on "a scatter plot on a
grid background" -- and the comparison the figure exists to make was written
down nowhere. No question about it could match anything.

Rejoining panels is only safe when the layout is unambiguous, so these tests
pin both directions: the matrix is rejoined, and things that merely sit near
each other are left exactly as they are.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from PIL import Image

from Core.pipelines.pdf_refiner import figure_merger


PANEL_PX = 28  # 14 units at 2 px/unit


def _panel(directory, index, x0, y0, width=14, height=14, *, colour=(10, 20, 30),
           block_type="image", caption=None, scale=2, page=4):
    path = os.path.join(directory, f"panel-{index}.jpg")
    Image.new("RGB", (width * scale, height * scale), colour).save(path)
    return {
        "type": block_type,
        "img_path": path,
        "image_caption": list(caption or []),
        "image_footnote": [],
        "content": f"panel {index} in isolation",
        "bbox": [x0 * 2, y0 * 2, (x0 + width) * 2, (y0 + height) * 2],
        "page_idx": page,
        "middle_json": {"type": "image", "bbox": [x0, y0, x0 + width, y0 + height]},
    }


def _grid(directory, rows, columns, *, pitch=20, **kwargs):
    panels = []
    index = 0
    for row in range(rows):
        for column in range(columns):
            panels.append(
                _panel(directory, index, 100 + column * pitch, 100 + row * pitch, **kwargs)
            )
            index += 1
    return panels


def _live(blocks):
    return [block for block in blocks if not block.get("invalid")]


class TheMatrixIsRejoined(unittest.TestCase):
    def test_a_three_by_five_matrix_becomes_one_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 3, 5)
            figure_merger(blocks)
            self.assertEqual(1, len(_live(blocks)))

    def test_the_figure_spans_every_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 3, 5)
            figure_merger(blocks)
            figure = _live(blocks)[0]
            # Union of the panel boxes, in the block's own bbox space: five
            # columns of 14 at a pitch of 20 span 100..194, three rows 100..154.
            self.assertEqual([200.0, 200.0, 388.0, 308.0], figure["bbox"])

    def test_every_panels_caption_is_kept_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks[0]["image_caption"] = ["CLASSIC SCT"]
            blocks[2]["image_caption"] = ["EDGEHD"]
            figure_merger(blocks)
            self.assertEqual(
                ["CLASSIC SCT", "EDGEHD"], _live(blocks)[0]["image_caption"]
            )

    def test_the_panel_descriptions_are_dropped_so_the_figure_is_described_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            figure_merger(blocks)
            self.assertEqual("", _live(blocks)[0]["content"])

    def test_the_figure_records_what_it_was_made_from(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            figure_merger(blocks)
            figure = _live(blocks)[0]
            self.assertIn(
                "bookrag.figure_merge.v1",
                str(figure.get("transformations", figure)),
            )

    def test_the_first_panel_in_reading_order_is_the_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 3)
            figure_merger(blocks)
            self.assertIs(blocks[0], _live(blocks)[0])


class TheCompositeShowsTheWholeFigure(unittest.TestCase):
    def test_the_composite_covers_the_whole_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            figure_merger(blocks)
            with Image.open(_live(blocks)[0]["img_path"]) as composite:
                # 34 units wide and tall at 2 px/unit.
                self.assertEqual((68, 68), composite.size)

    def test_each_panel_lands_where_it_sat_on_the_page(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = [
                _panel(directory, 0, 100, 100, colour=(255, 0, 0)),
                _panel(directory, 1, 120, 100, colour=(0, 255, 0)),
                _panel(directory, 2, 100, 120, colour=(0, 0, 255)),
                _panel(directory, 3, 120, 120, colour=(255, 255, 0)),
            ]
            figure_merger(blocks)
            # JPEG is lossy, so each panel is identified by which channels it is
            # bright in rather than by an exact triple.
            def channels(pixel):
                return tuple(value > 128 for value in pixel)

            with Image.open(_live(blocks)[0]["img_path"]) as composite:
                self.assertEqual((True, False, False), channels(composite.getpixel((5, 5))))
                self.assertEqual((False, True, False), channels(composite.getpixel((45, 5))))
                self.assertEqual((False, False, True), channels(composite.getpixel((5, 45))))
                self.assertEqual((True, True, False), channels(composite.getpixel((45, 45))))

    def test_the_composite_is_named_by_its_content(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _grid(directory, 2, 2)
            figure_merger(first)
            second = _grid(directory, 2, 2)
            figure_merger(second)
            self.assertEqual(
                _live(first)[0]["img_path"], _live(second)[0]["img_path"]
            )

    def test_a_missing_crop_leaves_the_panels_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            os.unlink(blocks[1]["img_path"])
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_panels_rendered_at_different_resolutions_are_not_pasted(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            # One panel re-rendered at a different scale means these crops do
            # not share a coordinate space.
            Image.new("RGB", (14 * 5, 14 * 5), (1, 2, 3)).save(blocks[3]["img_path"])
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))


class ThingsThatAreNotOneFigureStaySeparate(unittest.TestCase):
    def test_a_single_row_of_figures_is_left_alone(self):
        # Four figures side by side is exactly how a document lays out four
        # separate figures; merging them would fuse what it kept apart.
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 1, 4)
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_a_single_column_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 4, 1)
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_a_grid_with_a_hole_in_it_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 3)
            del blocks[4]
            figure_merger(blocks)
            self.assertEqual(5, len(_live(blocks)))

    def test_panels_of_different_sizes_are_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks[3] = _panel(directory, 3, 120, 120, width=40, height=40)
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_figures_far_apart_are_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2, pitch=60)
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_a_chart_is_never_absorbed_into_a_grid_of_images(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks[3]["type"] = "chart"
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_panels_on_different_pages_are_never_one_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks[2]["page_idx"] = 5
            blocks[3]["page_idx"] = 5
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))

    def test_text_between_panels_is_not_a_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks.append(
                {"type": "text", "text": "On-Axis", "bbox": [200, 200, 260, 210], "page_idx": 4}
            )
            figure_merger(blocks)
            live = _live(blocks)
            self.assertEqual(2, len(live))
            self.assertEqual({"image", "text"}, {block["type"] for block in live})

    def test_a_block_without_geometry_is_never_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            blocks = _grid(directory, 2, 2)
            blocks[1].pop("middle_json")
            figure_merger(blocks)
            self.assertEqual(4, len(_live(blocks)))


if __name__ == "__main__":
    unittest.main()
