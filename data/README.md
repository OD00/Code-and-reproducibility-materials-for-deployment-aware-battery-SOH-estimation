# Dataset layout

Download the public datasets and preserve their original folder names:

- [CALCE Battery Research Data](https://calce.umd.edu/data)
- [TJU Dataset 1 — Zhu et al.](https://doi.org/10.5281/zenodo.6379165)

```text
data/
  raw/
    CALCE/
      CS2_8/  CS2_21/  CS2_33/  CS2_34/
      CS2_35/ CS2_36/  CS2_37/  CS2_38/
    TJU/
      CY25-025_1/ CY25-05_1/ CY25-1_1/ CY35-05_1/ CY45-05_1/
  initial/CALCE/records/CALCE/       # generated
  initial/TJU/records/TJU_Dataset_1/ # generated
  cycle_dict_v1/                    # generated cycle dictionaries
  cycle_dict_qc_v2/                  # generated quality-controlled cycles
  model_input_v1/CALCE/             # generated 256-point arrays
  model_input_v1/TJU/               # TJU training inputs
  model_input_a1_v2/CALCE/          # CALCE training inputs and splits
```

The preparation scripts cover eight CALCE cells and five TJU groups. CALCE training selects CS2_35–38 after loading the complete prepared scope. TJU training selects `CY25-025_1` and `CY25-05_1`.

Run `python scripts/preprocess.py` from the repository root. Processing order:

1. `initial_preprocess_datasets_v2.py`: normalize raw records.
2. `build_cycle_dictionaries_v1.py`: pair charging/discharging cycles.
3. `process_cycle_quality_v2.py`: capacity reconstruction and quality control.
4. `build_model_input_v1.py`: 256-point resampling and strict windows.
5. `build_model_input_a1_v2.py`: CALCE gap-aware windows and purged splits.

Use `--stage` on the wrapper to run one step. Original audit/validation scripts remain in `preprocessing/`; use `--help` for arguments. Scalers are fitted during experiment loading rather than globally during preprocessing.
