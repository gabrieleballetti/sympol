from sympol.algorithms.classify_simplices import classify_simplices


def test_classify_simplices():
    """
    Classify all lattice triangles up to volume 6
    """
    # time the execution
    triangs = classify_simplices(2, 6)
    tot = [len(ts) for ts in triangs.values()]

    assert tot == [1, 1, 2, 3, 2, 3]
