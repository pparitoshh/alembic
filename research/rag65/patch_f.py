import ast, sys
p = sys.argv[1]; s = open(p).read()
R = [("""     a site missing from the indexes, the model comments on or follows unrelated Slurm excerpts).
\"\"\"""", """     a site missing from the indexes, the model comments on or follows unrelated Slurm excerpts).
  f  d's excerpts (general documentation index for non-Slurm questions) under e's header (Oct 25, opt-in).
\"\"\""""),
("""    if variant not in ('a', 'b', 'c', 'd', 'e'):
        raise ValueError('RAG_VARIANT must be unset, a, b, c, d or e')
    variant = 'b' if variant == 'e' else variant
""", """    if variant not in ('a', 'b', 'c', 'd', 'e', 'f'):
        raise ValueError('RAG_VARIANT must be unset, a, b, c, d, e or f')
    variant = {'e': 'b', 'f': 'd'}.get(variant, variant)
"""),
("""    if os.environ.get('RAG_VARIANT', '') == 'e':""", """    if os.environ.get('RAG_VARIANT', '') in ('e', 'f'):"""),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
ast.parse(s); open(p + '.new', 'w').write(s); print('ok')
