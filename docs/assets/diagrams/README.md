# Diagrams

Editable diagram sources and their rendered assets live together here.

`pipeline-flow.zh.dot` is the styled Graphviz source for
`pipeline-flow.zh.svg` and `pipeline-flow.zh.png`; `pipeline-flow.zh.mmd` is the
compact Mermaid equivalent used when Graphviz is unavailable. Documentation
should embed the SVG and link to an editable source when a diagram changes.

Regenerate the Graphviz outputs with:

```bash
dot -Tsvg docs/assets/diagrams/pipeline-flow.zh.dot \
  -o docs/assets/diagrams/pipeline-flow.zh.svg
dot -Tpng docs/assets/diagrams/pipeline-flow.zh.dot \
  -o docs/assets/diagrams/pipeline-flow.zh.png
```
