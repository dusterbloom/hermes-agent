"""Tests for markdown rendering in AI responses (fixes raw **bold** display)."""
import pytest
from pathlib import Path

_CLI_PY = Path(__file__).parent.parent / "cli.py"


class TestMarkdownImport:
    """Verify rich.markdown.Markdown is imported and available."""

    def test_markdown_imported_in_cli(self):
        """cli.py must import Markdown from rich.markdown."""
        source = _CLI_PY.read_text()
        assert "from rich.markdown import Markdown" in source, \
            "cli.py does not import rich.markdown.Markdown"

    def test_rich_markdown_renders_bold(self):
        """Rich Markdown must convert **bold** to styled text, not raw asterisks."""
        from rich.console import Console
        from rich.markdown import Markdown
        from io import StringIO
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, color_system="truecolor")
        console.print(Markdown("This is **bold** text"))
        output = buf.getvalue()
        assert "**bold**" not in output, \
            f"Raw markdown asterisks found in rendered output: {output!r}"


class TestNonStreamingUsesMarkdown:
    """The non-streaming Panel path must use Markdown(), not _rich_text_from_ansi()."""

    def test_panel_path_uses_markdown_not_raw_text(self):
        """In the non-streaming response branch, Panel content must be Markdown."""
        source = _CLI_PY.read_text()
        # Find the Panel() call in the non-streaming path
        # It should use Markdown(response), not _rich_text_from_ansi(response)
        import re
        panel_calls = re.findall(r'Panel\(\s*\n?\s*(\w+)\(response\)', source)
        assert any("Markdown" in call for call in panel_calls), \
            f"Panel content uses {panel_calls} instead of Markdown(response)"


class TestStreamingReRender:
    """After streaming completes, raw output should be replaced with rendered markdown."""

    def test_stream_line_counter_exists(self):
        """The streaming path must track line count for re-render clearing."""
        source = _CLI_PY.read_text()
        assert "_stream_line_count" in source, \
            "No _stream_line_count variable — streaming re-render won't work"

    def test_stream_full_text_accumulator_exists(self):
        """The streaming path must accumulate full text for markdown re-render."""
        source = _CLI_PY.read_text()
        assert "_stream_full_text" in source, \
            "No _stream_full_text accumulator — streaming re-render won't work"

    def test_already_streamed_branch_does_not_just_pass(self):
        """The already_streamed branch must re-render, not just 'pass'."""
        source = _CLI_PY.read_text()
        # Find the already_streamed branch — it should NOT just be 'pass'
        import re
        match = re.search(r'elif already_streamed:\s*\n\s*(pass|# Token)', source)
        assert match is None or "pass" not in match.group(1), \
            "already_streamed branch still just does 'pass' — no markdown re-render"


class TestStreamingEraseUsesZeroWidthEscape:
    """The already_streamed erase sequence must not go through sys.stdout.write.

    Inside prompt_toolkit's patch_stdout() context, sys.stdout is replaced by
    StdoutProxy which mangles ANSI escape bytes (replaces \\x1b with '?').
    Cursor-movement sequences such as \\033[NA\\033[J must be sent via
    ZeroWidthEscape or write_raw so they bypass StdoutProxy's sanitisation.
    """

    @staticmethod
    def _already_streamed_branch(source: str) -> str:
        """Extract the text of the elif already_streamed: block from cli.py source."""
        import re
        # Grab from the 'elif already_streamed:' line to the next 'else:' or 'elif '
        # at the same indentation level.  We take a generous slice so the checks
        # are reliable regardless of minor refactors.
        match = re.search(
            r'(elif already_streamed:.*?)(?=\n\s{16}(?:elif |else:))',
            source,
            re.DOTALL,
        )
        assert match is not None, \
            "Could not locate 'elif already_streamed:' block in cli.py"
        return match.group(1)

    def test_no_raw_stdout_write_in_already_streamed_branch(self):
        """The already_streamed branch must NOT call sys.stdout.write.

        StdoutProxy.write() replaces \\x1b with '?', so any raw ANSI sequence
        written via sys.stdout.write() inside a patch_stdout() context will be
        silently corrupted.  This test will FAIL until the fix is applied.
        """
        source = _CLI_PY.read_text()
        branch = self._already_streamed_branch(source)
        assert "sys.stdout.write" not in branch, (
            "already_streamed branch still uses sys.stdout.write — "
            "ANSI escape sequences will be mangled by prompt_toolkit StdoutProxy"
        )

    def test_uses_zero_width_escape_or_write_raw(self):
        """The already_streamed branch must use ZeroWidthEscape or write_raw.

        These are the two prompt_toolkit-safe ways to emit raw ANSI sequences
        without going through StdoutProxy's byte sanitisation.
        This test will FAIL until the fix is applied.
        """
        source = _CLI_PY.read_text()
        branch = self._already_streamed_branch(source)
        uses_safe_api = "ZeroWidthEscape" in branch or "write_raw" in branch
        assert uses_safe_api, (
            "already_streamed branch neither uses ZeroWidthEscape nor write_raw — "
            "cursor-movement ANSI sequences will be corrupted by StdoutProxy"
        )
