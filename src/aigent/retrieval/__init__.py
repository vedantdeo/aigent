"""Retrieval: documents in, chunks out, embeddings over them, a ranked list back.

    chunk    Document -> Chunk: fixed, fixed-with-overlap, by sentence, by heading
    embed    Chunk.text -> vectors, from a local model
    store    vectors -> a ranked list of chunk ids, brute-force cosine

Nothing here calls the Anthropic API, so every retrieval number is free to re-measure.
"""
