from typer.testing import CliRunner
from findex.cli import app

runner = CliRunner()

def test_cli_search_json():
    result = runner.invoke(app, ["search", "dummy_path", "apple", "--json"])
    assert result.exit_code == 0
    assert "[{" in result.stdout
