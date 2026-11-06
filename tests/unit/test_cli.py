from importlib.metadata import version

from typer.testing import CliRunner

from citerag.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0 and version("citerag") in result.stdout
