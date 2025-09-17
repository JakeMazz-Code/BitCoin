from bdssim.playbooks import get_playbook_by_key, load_playbook_index, reduce_playbook


def test_playbook_index_loads() -> None:
    index = load_playbook_index("configs/playbooks/index.yaml")
    assert index, "playbook index is empty"
    keys = {pb.key for pb in index}
    assert "transatlantic" in keys
    pb = get_playbook_by_key("transatlantic")
    assert pb is not None
    waves = reduce_playbook(pb)
    assert waves[0]["countries"]
