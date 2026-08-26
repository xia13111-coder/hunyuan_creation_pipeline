# Validated local results

Validated material deliveries may be archived under `<asset>/<version>/`.
Generated results are ignored by Git; only this note is versioned. Keep new runs
in `../var/runs/` until all delivery checks pass.

Generated releases must use relative USD/JSON references and must not be
committed with machine-local paths. View a validated local release by setting
`DELIVERY_DIR` when running `../web/result_viewer/serve.sh`.
