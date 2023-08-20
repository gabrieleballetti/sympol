from sympol.ehrhart import is_valid_h_star_vector


def test_is_valid_h_star_vector():
    """Test is_valid_h_star_vector function."""

    # ordered so that all the failure cases are covered, in order
    assert not is_valid_h_star_vector((1, 1, 1.0))
    assert not is_valid_h_star_vector((2, 1, 1))
    assert not is_valid_h_star_vector((1, -1, 1))
    assert not is_valid_h_star_vector((1, 1, 2))
    assert not is_valid_h_star_vector((1, 1, 1, 2, 1))
    assert not is_valid_h_star_vector((1, 1, 0, 1, 0))
    assert not is_valid_h_star_vector((1, 2, 1, 1, 2))
    assert not is_valid_h_star_vector((1, 2, 1, 1, 0))

    assert is_valid_h_star_vector((1, 1, 1))
    assert is_valid_h_star_vector((1, 1, 1, 0))
