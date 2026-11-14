from math_posttrain.splits import assert_disjoint, carve_dev


def test_carve_dev_is_deterministic_and_disjoint() -> None:
    ids = [str(index) for index in range(20)]
    train, dev = carve_dev(ids, dev_size=5, seed=42)
    assert len(train) == 15
    assert len(dev) == 5
    assert_disjoint(train, dev)
    train_again, dev_again = carve_dev(ids, dev_size=5, seed=42)
    assert (train_again, dev_again) == (train, dev)


def test_different_seed_changes_dev_membership() -> None:
    ids = [str(index) for index in range(20)]
    _, dev_a = carve_dev(ids, dev_size=5, seed=1)
    _, dev_b = carve_dev(ids, dev_size=5, seed=2)
    assert dev_a != dev_b


def test_carve_dev_rejects_invalid_size() -> None:
    try:
        carve_dev(["1", "2"], dev_size=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_assert_disjoint_rejects_overlap() -> None:
    try:
        assert_disjoint(["1", "2"], ["2", "3"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
