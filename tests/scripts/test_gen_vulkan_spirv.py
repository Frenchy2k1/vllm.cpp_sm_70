"""Unit tests for scripts/gen-vulkan-spirv.py (BACKEND-VULKAN, VK-A1).

The SpecId parser is the piece worth testing directly: it walks raw SPIR-V, and
the two ways a naive scanner breaks (mistaking another decoration for SpecId, and
failing to step over unrelated instructions by word count) are both silent — they
produce a wrong ID list, which becomes a wrong VkSpecializationInfo, which Vulkan
accepts without complaint and turns into wrong numbers.
"""

import importlib.util
import pathlib
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "gen_vulkan_spirv", ROOT / "scripts" / "gen-vulkan-spirv.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def module(*instructions: "tuple[int, list[int]]") -> bytes:
    """Build a minimal SPIR-V blob: the 5 header words, then the instructions."""
    words = [0x07230203, 0x00010300, 0, 1, 0]
    for opcode, operands in instructions:
        words.append(((len(operands) + 1) << 16) | opcode)
        words.extend(operands)
    return b"".join(struct.pack("<I", w) for w in words)


class SpecIdParsing(unittest.TestCase):
    def test_no_decorations_yields_empty(self):
        self.assertEqual(gen.spec_ids(module()), [])

    def test_extracts_spec_ids_sorted_and_deduped(self):
        blob = module(
            (gen.SPIRV_OP_DECORATE, [10, gen.SPIRV_DECORATION_SPEC_ID, 3]),
            (gen.SPIRV_OP_DECORATE, [11, gen.SPIRV_DECORATION_SPEC_ID, 0]),
            (gen.SPIRV_OP_DECORATE, [12, gen.SPIRV_DECORATION_SPEC_ID, 3]),
        )
        self.assertEqual(gen.spec_ids(blob), [0, 3])

    def test_ignores_other_decorations(self):
        # Decoration 2 is Block, not SpecId. Collecting it would invent an ID.
        blob = module((gen.SPIRV_OP_DECORATE, [10, 2, 0]))
        self.assertEqual(gen.spec_ids(blob), [])

    def test_steps_over_unrelated_opcodes_by_word_count(self):
        # OpName (5) carries a literal string; a scanner that advanced by a fixed
        # stride would land mid-instruction and misread the following decoration.
        blob = module(
            (5, [10, 0x6E69616D, 0]),
            (gen.SPIRV_OP_DECORATE, [10, gen.SPIRV_DECORATION_SPEC_ID, 7]),
        )
        self.assertEqual(gen.spec_ids(blob), [7])

    def test_ignores_too_short_decorate(self):
        # OpDecorate with no literal operand cannot be a SpecId; reading operand 3
        # would run off the end of the instruction.
        blob = module((gen.SPIRV_OP_DECORATE, [10, gen.SPIRV_DECORATION_SPEC_ID]))
        self.assertEqual(gen.spec_ids(blob), [])


class CommittedArtifact(unittest.TestCase):
    def test_module_table_carries_the_spec_id_columns(self):
        """The committed header must expose the specialization columns at all.

        Read from the COMMITTED header rather than by recompiling, so this runs
        with no shader toolchain — the same property that lets the C++ gate check
        the artifact on a box with no Vulkan.

        assertIn is avoided throughout: the haystack is a ~110 KB generated header
        and assertIn prints it in full on failure, burying the message.
        """
        header = (ROOT / "src/vt/vulkan/vulkan_spirv.h").read_text()
        for field in ("const uint32_t* spec_ids;", "size_t spec_id_count;"):
            self.assertTrue(field in header,
                            f"SpirvModule is missing {field!r}; "
                            f"re-run scripts/gen-vulkan-spirv.py")

    def test_vt_cast_declares_its_dtype_pair(self):
        """vt_cast is the backend's first variant axis: src and dst dtype.

        One committed module serves every (src, dst) pair, specialized at pipeline
        creation, instead of llama.cpp's module-per-#define. The workgroup size is
        deliberately NOT such a constant: local_size_x_id emits LocalSize 1 1 1 at
        the vulkan1.1 target (see vt_common.glsl).
        """
        header = (ROOT / "src/vt/vulkan/vulkan_spirv.h").read_text()
        self.assertTrue("kSpecIds_vt_cast[]" in header,
                        "vt_cast no longer declares specialization constants; "
                        "re-run scripts/gen-vulkan-spirv.py")

    def test_other_shaders_declare_none_yet(self):
        """Stated as a fact so the next shader to take a constant flips this test.

        Whoever adds one is then forced to supply matching values at its dispatch,
        which GetPipeline checks by count against the declared SpecIds.
        """
        header = (ROOT / "src/vt/vulkan/vulkan_spirv.h").read_text()
        for src in sorted((ROOT / "src/vt/vulkan/shaders").glob("*.comp")):
            if src.stem == "vt_cast":
                continue
            with self.subTest(shader=src.name):
                self.assertTrue(
                    f"kSpecIds_{src.stem}[]" not in header,
                    f"{src.name} now declares a specialization constant; update "
                    f"this test and pass its value at the dispatch")


if __name__ == "__main__":
    unittest.main()
