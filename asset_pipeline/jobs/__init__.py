"""Primitive jobs grouped by execution boundary.

``cad`` validates STEP/STP and invokes CAD Converter; the sibling
``asset_pipeline.visual_materials`` package builds the Look USD; ``isaac``
owns Physics and collection; ``delivery`` performs the
post-collection fail-closed validation. End-to-end ordering belongs in
``asset_pipeline.manual_cad``, not in this package.
"""
