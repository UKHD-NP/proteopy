"""
Tests for proteopy.utils.parsers.parse_stat_test_varm_slot function.
"""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from proteopy.utils.parsers import parse_stat_test_varm_slot


class TestParseStatTestVarmSlot:
    """Tests for parse_stat_test_varm_slot function."""

    def test_parse_welch_two_group_no_layer(self):
        """Test parsing a Welch's t-test two-group slot without layer."""
        slot = "welch;condition;treated_vs_control"
        result = parse_stat_test_varm_slot(slot)

        assert result["test_type"] == "welch"
        assert result["test_type_label"] == "Welch's t-test"
        assert result["group_by"] == "condition"
        assert result["design"] == "treated_vs_control"
        assert result["design_label"] == "treated vs control"
        assert result["layer"] is None

    def test_parse_ttest_two_sample_no_layer(self):
        """Test parsing a two-sample t-test slot without layer."""
        slot = "ttest_two_sample;cell_type;A_vs_B"
        result = parse_stat_test_varm_slot(slot)

        assert result["test_type"] == "ttest_two_sample"
        assert result["test_type_label"] == "Two-sample t-test"
        assert result["group_by"] == "cell_type"
        assert result["design"] == "A_vs_B"
        assert result["design_label"] == "A vs B"
        assert result["layer"] is None

    def test_parse_one_vs_rest_no_layer(self):
        """Test parsing a one-vs-rest slot without layer."""
        slot = "welch;condition;treated_vs_rest"
        result = parse_stat_test_varm_slot(slot)

        assert result["test_type"] == "welch"
        assert result["group_by"] == "condition"
        assert result["design"] == "treated_vs_rest"
        assert result["design_label"] == "treated vs rest"
        assert result["layer"] is None

    def test_parse_with_layer(self):
        """Test parsing a slot with a layer specified."""
        slot = "welch;condition;treated_vs_control;raw_intensities"
        result = parse_stat_test_varm_slot(slot)

        assert result["test_type"] == "welch"
        assert result["group_by"] == "condition"
        assert result["design"] == "treated_vs_control"
        assert result["layer"] == "raw_intensities"

    def test_layer_resolution_with_adata(self):
        """Test that layer is resolved to original name when adata provided."""
        # Create AnnData with a layer that has spaces
        proteins = ["PROT_A", "PROT_B"]
        adata = AnnData(
            np.arange(4).reshape(2, 2),
            var=pd.DataFrame(index=proteins)
        )
        adata.var["protein_id"] = proteins
        adata.layers["Raw Intensities"] = np.arange(4).reshape(2, 2)

        slot = "welch;condition;treated_vs_control;Raw_Intensities"
        result = parse_stat_test_varm_slot(slot, adata=adata)

        assert result["layer"] == "Raw Intensities"

    def test_invalid_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_stat_test_varm_slot("")

    def test_invalid_none_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            parse_stat_test_varm_slot(None)

    def test_invalid_format_wrong_part_count_raises(self):
        """Test that wrong number of parts raises ValueError."""
        with pytest.raises(ValueError, match="must have format"):
            parse_stat_test_varm_slot("welch_condition_treated_vs_control")

    def test_invalid_format_too_many_parts_raises(self):
        """Test that too many parts raises ValueError."""
        with pytest.raises(ValueError, match="must have format"):
            parse_stat_test_varm_slot("welch;condition;A_vs_B;layer;extra")

    def test_invalid_test_type_raises(self):
        """Test that unsupported test type raises ValueError."""
        slot = "unknown_test;condition;A_vs_B"
        with pytest.raises(ValueError, match="not supported"):
            parse_stat_test_varm_slot(slot)

    def test_missing_group_by_raises(self):
        """Test that missing group_by raises ValueError."""
        slot = "welch;;A_vs_B"
        with pytest.raises(ValueError, match="missing the group_by"):
            parse_stat_test_varm_slot(slot)

    def test_missing_design_raises(self):
        """Test that missing design raises ValueError."""
        slot = "welch;condition;"
        with pytest.raises(ValueError, match="missing the design"):
            parse_stat_test_varm_slot(slot)

    def test_invalid_design_format_raises(self):
        """Test that design without _vs_ raises ValueError."""
        slot = "welch;condition;invalid_design"
        with pytest.raises(ValueError, match="Design must use"):
            parse_stat_test_varm_slot(slot)

    def test_design_missing_group1_raises(self):
        """Test that design with empty group1 raises ValueError."""
        slot = "welch;condition;_vs_B"
        with pytest.raises(ValueError, match="missing group labels"):
            parse_stat_test_varm_slot(slot)

    def test_design_missing_group2_raises(self):
        """Test that design with empty group2 raises ValueError."""
        slot = "welch;condition;A_vs_"
        with pytest.raises(ValueError, match="missing group labels"):
            parse_stat_test_varm_slot(slot)

    def test_vs_rest_missing_group_raises(self):
        """Test that _vs_rest without group raises ValueError."""
        slot = "welch;condition;_vs_rest"
        with pytest.raises(ValueError, match="missing the group label"):
            parse_stat_test_varm_slot(slot)

    def test_sanitized_group_names(self):
        """Test parsing with sanitized group names containing underscores."""
        slot = "welch;sample_condition;Group_A_vs_Group_B"
        result = parse_stat_test_varm_slot(slot)

        assert result["group_by"] == "sample_condition"
        assert result["design"] == "Group_A_vs_Group_B"
        assert result["design_label"] == "Group A vs Group B"
