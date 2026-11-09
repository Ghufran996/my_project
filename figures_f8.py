"""Figure 6 (Sobol indices) is produced by figures_final.py (data_f8 writes the CSV, draw_fig6 draws it); this wrapper keeps the old entry name. Run after uq_final_summary.py sobol and sobolbeta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_final import data_f8, draw_fig6  # noqa: E402

if __name__ == "__main__":
    data_f8()
    print(draw_fig6())
