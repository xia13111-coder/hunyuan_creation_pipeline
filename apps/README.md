# Applications

Standalone user interfaces live here. They read pipeline results but are not
part of the pipeline runtime.

- `material_audit_web/`: local DTN100 material and alignment viewer. It is a
  separate repository and is excluded from the public source archive because it
  contains reference photographs and generated diagnostic data. Review privacy and
  licenses before publishing it.

Frontend dependencies and build output (`node_modules/`, `.next/`, `dist/`) are
rebuildable and are not source artifacts.
