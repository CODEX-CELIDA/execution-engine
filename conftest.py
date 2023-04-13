from glob import glob

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "")
    for fixture_file in glob("tests/fixtures/[!__]*.py", recursive=True)
]
