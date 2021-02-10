import pytest

import numpy as np
import pandas as pd
import pandas.util.testing as pdt
from pandas.tests.indexing.common import Base

#: List of pairs of dloc vs loc equivalent slicing arguments
ACCESSORS_1D_EQUIVALENT = [
    ({"level0": "B"}, "B"),
    ({"level0": ["A"]}, ["A"]),
    ({"level0": slice("A", "B")}, slice("A", "B")),
    ({"level1": "A"}, (slice(None), "A")),
    ({"level1": ["A", "B"]}, (slice(None), ["A", "B"])),
    ({"level0": "A", "level1": "A"}, ("A", "A")),
    (slice(None), slice(None)),
    ({"level1": slice("A", "B")}, (slice(None), slice("A", "B"))),
]


@pytest.fixture
def test_frame():
    return pd.DataFrame(
        np.arange(0, 16).reshape((4, 4)),
        index=pd.MultiIndex.from_tuples(
            [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A")], names=["level0", "level1"]
        ),
        columns=pd.MultiIndex.from_tuples(
            [("A", "A"), ("A", "B"), ("A", "C"), ("B", "A")], names=["level0", "level1"]
        ),
    )


def assert_pandas_equal(left, right):
    if isinstance(left, pd.DataFrame):
        pdt.assert_frame_equal(left, right)
    elif isinstance(left, pd.Series):
        pdt.assert_series_equal(left, right)
    else:
        np.all(np.isclose(left, right))


class TestDLoc(Base):
    @pytest.mark.parametrize("axis", (0, 1))
    @pytest.mark.parametrize(("slice_dloc", "slice_loc"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_axis_access(self, test_frame, axis, slice_dloc, slice_loc):
        assert_pandas_equal(
            test_frame.dloc(axis)[slice_dloc],
            test_frame.loc(axis)[slice_loc],
        )

    @pytest.mark.parametrize(("slice_dloc", "slice_loc"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_tuple_access_no_slice(self, test_frame, slice_dloc, slice_loc):
        assert_pandas_equal(
            test_frame.dloc[slice_dloc, :], test_frame.loc[slice_loc, :]
        )

    @pytest.mark.parametrize(("slice_dloc0", "slice_loc0"), ACCESSORS_1D_EQUIVALENT)
    @pytest.mark.parametrize(("slice_dloc1", "slice_loc1"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_tuple_access_multiple_axis(
        self, test_frame, slice_dloc0, slice_dloc1, slice_loc0, slice_loc1
    ):
        if slice_loc0 == "A" and slice_loc1 == "A":
            pytest.skip(
                "loc scalar access this way doesnt make sense compared to dloc usage"
            )
        dloc_out = test_frame.dloc[slice_dloc0, slice_dloc1]
        loc_out = test_frame.loc[slice_loc0, slice_loc1]
        assert_pandas_equal(dloc_out, loc_out)

    @pytest.mark.parametrize("axis", (0, 1))
    @pytest.mark.parametrize(("slice_dloc", "slice_loc"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_setitem_axis(self, test_frame, axis, slice_dloc, slice_loc):
        dloc_to_set, loc_to_set = test_frame.copy(), test_frame.copy()
        dloc_to_set.dloc(axis=axis)[slice_dloc] = 99.0
        loc_to_set.loc(axis=axis)[slice_loc] = 99.0
        assert_pandas_equal(dloc_to_set, loc_to_set)

    @pytest.mark.parametrize(("slice_dloc", "slice_loc"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_setitem_tuple(self, test_frame, slice_dloc, slice_loc):
        dloc_to_set, loc_to_set = test_frame.copy(), test_frame.copy()
        dloc_to_set.dloc[slice_dloc, :] = 99.0
        loc_to_set.loc[slice_loc, :] = 99.0
        assert_pandas_equal(dloc_to_set, loc_to_set)

    @pytest.mark.parametrize(("slice_dloc0", "slice_loc0"), ACCESSORS_1D_EQUIVALENT)
    @pytest.mark.parametrize(("slice_dloc1", "slice_loc1"), ACCESSORS_1D_EQUIVALENT)
    def test_dloc_setitem_tuple_multiple_axis(
        self, test_frame, slice_dloc0, slice_dloc1, slice_loc0, slice_loc1
    ):
        if slice_loc0 == "A" and slice_loc1 == "A":
            pytest.skip(
                "loc scalar access this way doesnt make sense compared to dloc usage"
            )
        dloc_to_set, loc_to_set = test_frame.copy(), test_frame.copy()
        dloc_to_set.dloc[slice_dloc0, slice_dloc1] = 99.0
        loc_to_set.loc[slice_loc0, slice_loc1] = 99.0
        assert_pandas_equal(dloc_to_set, loc_to_set)
