from llamacpp_stack import cli


def test_fit_flags_single_dash():
    cmd = []
    cli._append_llama_server_flag(cmd, "fit", True)
    assert "-fit" in cmd
    assert "--fit" not in cmd

    cmd = []
    cli._append_llama_server_flag(cmd, "fitt", 2048)
    assert "-fitt" in cmd
    assert "--fitt" not in cmd

    cmd = []
    cli._append_llama_server_flag(cmd, "fitc", True)
    assert "-fitc" in cmd
    assert "--fitc" not in cmd


def test_normalize_server_overrides_preserves_fit_keys():
    normalized = cli.normalize_server_overrides({"fit": "off", "fitt": "1024", "fitc": "131072"})
    assert normalized["fit"] is False
    assert normalized["fitt"] == 1024
    assert normalized["fitc"] == 131072
