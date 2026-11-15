# Trust-rule figures

**Figure T1. Withholding the targets a tool is least likely to get right (n = 218 held-out targets).** Risk is the share of kept targets where the tool's top-10 contains no true pocket residue. Targets are ranked by the trust score (L2 logistic on the tool's own signal plus cross-tool agreement, fitted on 846 training targets) or by the tool's own top score, and the top fraction is kept. Dashed: keeping everything.

**Figure T2. The trust score is calibrated (held-out).** Predicted probability against observed accuracy in ten equal-width bins, marker area proportional to bin size; the diagonal is perfect calibration. The calibrator (Platt or isotonic, whichever had the lower out-of-fold error) was fitted on out-of-fold training predictions and applied once to the test split.
