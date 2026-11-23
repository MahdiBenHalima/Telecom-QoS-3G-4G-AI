import pandas as pd
from training.train_3g import label_3g, load_3g_data, make_pipeline, train


def test_original_kpi_labels():
    frame = pd.DataFrame({
        "rscp_active": [-70, -85, -105],
        "ecno_active": [-5, -9, -17],
    })
    assert list(label_3g(frame)) == [
        "Connexion recommandée",
        "Connexion possible mais instable",
        "Connexion faible / non recommandée",
    ]


def test_train_and_export_with_synthetic_fixture(tmp_path):
    # Fabricated measurements ONLY; not the private operator dataset.
    rows = []
    for i in range(60):
        group = i % 3
        rscp, ecno = [(-70, -5), (-86, -10), (-106, -17)][group]
        rows.append({
            "Time": f"12:00:{i:02d}.000",
            "Band (active)": "Band 1",
            "Channel number (active)": "10564",
            "Scrambling code (active)": str(i % 15),
            "RSCP (active)": str(rscp),
            "Ec/N0 (active)": str(ecno),
            "RSCP (detected)": "",
            "Scrambling code (detected)": "",
        })
    source = tmp_path / "DT2.csv"
    pd.DataFrame(rows).to_csv(source, sep=";", index=False)
    clean = load_3g_data(source)
    assert len(clean) == 60
    out = tmp_path / "models"
    results = train(source, out)
    assert len(results) == 3
    assert (out / "rf_3g_pipeline.joblib").exists()
    assert (out / "label_encoder_3g.joblib").exists()
    assert (out / "evaluation_3g.json").exists()
