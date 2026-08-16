# Retrieval branch finalization implementation plan

1. Update the BM25F component defaults to B3: `k1=1.2`, `b=0.5`,
   `title_b=0.75`, `title_boost=3.0`, and `top_k=10`.
2. Update the benchmark factory and CLI defaults to BM25 B3 parameters and BGE
   `batch_size=8`; keep vector V2 as the default.
3. Update tests and component documentation to match the new defaults.
4. Add a compact tracked experiment report sourced from the frozen full-run
   summaries and paired-bootstrap JSON files.
5. Rebuild the ignored aggregate CSV from all six paired comparison files.
6. Add `.idea/` to the repository ignore rules and remove the whitespace warning.
7. Run the complete test suite and repository checks, then create a local commit.
