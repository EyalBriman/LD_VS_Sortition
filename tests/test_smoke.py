from pathlib import Path

from ld_vs_sortition.experiments import experiment1, experiment2, loss_bound


def test_loss_bound_smoke(tmp_path: Path):
    results, summary = loss_bound.run_and_save(
        tmp_path / "loss",
        n_list=[5],
        n_instances=2,
        graphs_per_instance=20,
        base_seed=1,
        batch_size=20,
    )
    assert len(results) == 2
    assert len(summary) == 1
    assert (tmp_path / "loss" / "loss_bound_summary.csv").exists()


def test_experiment1_smoke(tmp_path: Path):
    mean_df, median_df, identity = experiment1.run_and_save(
        tmp_path / "exp1",
        n_list=[8],
        gamma_list=[0.5, 1.0],
        n_mc=20,
        theorem_n_list=[8],
        theorem_gamma_list=[1.0],
        theorem_graphs=30,
        theorem_mc=20,
        q_graphs=30,
        base_seed=2,
    )
    assert not mean_df.empty
    assert not median_df.empty
    assert not identity.empty


def test_experiment2_smoke(tmp_path: Path):
    mean_df, median_df = experiment2.run_and_save(
        tmp_path / "exp2",
        n_list=[12],
        gamma_list=[1.0],
        k_list=[1, 3],
        n_mc=2,
        base_seed=3,
        sbm_configs=[
            {
                "label": "Test SBM (B=2, p_in=0.4, p_out=0.1)",
                "blocks": 2,
                "p_in": 0.4,
                "p_out": 0.1,
            }
        ],
    )
    assert len(mean_df) == 4
    assert len(median_df) == 4
    assert (tmp_path / "exp2" / "exp2_summary.csv").exists()
