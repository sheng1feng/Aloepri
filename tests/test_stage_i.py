import torch

from src.stage_i_vllm import build_phase2_feasibility_summary, build_stage_i_manifest, summarize_token_partitions


class _MockTokenizer:
    vocab_size = 6
    all_special_ids = [0, 1, 8]


def test_build_stage_i_manifest_has_expected_layout() -> None:
    manifest = build_stage_i_manifest()
    assert manifest["format"] == "stage_i_vllm_v1"
    assert manifest["server_dir"] == "server"
    assert manifest["client_dir"] == "client"
    assert "model.safetensors" in manifest["server_files"]
    assert "client_secret.pt" in manifest["client_files"]


def test_summarize_token_partitions_marks_special_and_tail_fixed() -> None:
    perm_vocab = torch.tensor([0, 1, 3, 4, 5, 2, 6, 7], dtype=torch.long)
    summary = summarize_token_partitions(
        tokenizer=_MockTokenizer(),
        model_vocab_size=8,
        perm_vocab=perm_vocab,
    )
    assert summary["perm_is_valid"] is True
    assert summary["special_ids_fixed"] is True
    assert summary["tail_rows_fixed"] is True
    assert summary["tail_row_count"] == 2


def test_phase2_feasibility_flags_rmsnorm_as_not_vllm_ready() -> None:
    feasibility = build_phase2_feasibility_summary()
    component_map = {item["component"]: item for item in feasibility["components"]}
    assert component_map["Stage A vocab permutation"]["status"] == "feasible_now"
    assert component_map["RMSNorm fused path"]["status"] == "not_vllm_ready"
