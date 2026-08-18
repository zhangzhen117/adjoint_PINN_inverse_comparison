"""The two ablation studies of Appendix B.

b1_optimizer     Appendix B.1, Table 6 -- Burgers PINN under {Adam, SOAP, SSBroyden}
b2_architecture  Appendix B.2, Table 7 -- Allen-Cahn PINN width, depth, activation,
                 and Fourier input encoding

Both are driven exactly like the sweeps in ``sweeps/``: ``rows()`` fixes the
configuration order and ``--row N`` runs one of them.
"""
