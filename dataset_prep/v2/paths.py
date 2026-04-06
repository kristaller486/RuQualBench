from pathlib import Path


V2_DATASET_PREP_DIR = Path(__file__).resolve().parent
REPO_ROOT = V2_DATASET_PREP_DIR.parent.parent
SCRIPTS_DIR = V2_DATASET_PREP_DIR / "scripts"
CONFIGS_DIR = V2_DATASET_PREP_DIR / "configs"
SOURCES_DIR = V2_DATASET_PREP_DIR / "sources"
SELECTIONS_DIR = V2_DATASET_PREP_DIR / "selections"
OUTPUTS_DIR = V2_DATASET_PREP_DIR / "outputs"
FINAL_DIR = V2_DATASET_PREP_DIR / "final"
PROMPTS_DIR = REPO_ROOT / "prompts"
