"""Training utilities for the patched-DA3 + Mamba-3 attention pipeline.

Phase 1 — `phase1.train`: distill against DA3-LARGE teacher.
Phase 2 — `phase2.train`: GT-supervised head adaptation.
Phase 3 — `phase3.train`: full-unfreeze co-adaptation.
Loss   — `da3_loss.da3_paper_loss`: DA3 § 3.3 (L_D + L_M + L_grad + L_P + β·L_C).
Data   — `multi_view.multi_view_iterator`: ETH3D / HiRoom / 7Scenes scenes.
"""
