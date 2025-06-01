from sympol.polytope import Polytope


def test_idp_non_log_concave():
    """
    Test examples of idp polytopes with non-log-concave h*-vectors from the paper
    J. Hofscheier, V. Kurylenko, and B. Nill.
    """
    points = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, -1, -1, -1, -1, 0, 0],
        [-1, -1, 0, 0, 0, -1, -1],
    ]
    p1 = Polytope(points)

    assert p1.is_idp
    assert not p1.has_log_concave_h_star_vector
    assert p1.is_ehrhart_positive

    points = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ]
    p2 = Polytope(points)

    assert p2.is_idp
    assert not p2.has_log_concave_h_star_vector
    assert p2.is_ehrhart_positive


def test_extra_idp_not_lc():
    """
    Test other examples
    """

    # d = 10
    # points10 = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    #     [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    # ]

    # p10 = Polytope(points10)
    # assert p10.is_idp
    # assert not p10.has_log_concave_h_star_vector
    # assert p10.is_ehrhart_positive

    # d = 9
    points9 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1, 1, 1, 0],
    ]

    p9 = Polytope(points9)
    assert p9.is_idp
    assert not p9.has_log_concave_h_star_vector
    assert p9.is_ehrhart_positive
