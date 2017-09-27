import epi

def test_parity():
    for pf in [epi.parity1, epi.parity2]:
        assert pf(1) == 1
        assert pf(2) == 1
        assert pf(3) == 0
        assert pf(4) == 1
        assert pf(123456789) == 0
