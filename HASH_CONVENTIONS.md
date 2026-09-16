# Hash scopes in the sanitized derivative

1. `MANIFEST.json` binds every **public packaged file** except itself.
2. `SOURCE_MAP.json` binds the selected original file digest and the final public derivative digest,
   and marks whether bytes are unchanged. A source digest is not claimed to be a public-file digest.
3. `paper/FIGURE_BINDING.json` is regenerated against the **public** result JSON bytes. Its hashes
   can be checked directly in this snapshot.
4. The two `CURRENT_RESULTS.json` locators bind their `accepted_current_result_sha256` to the public
   sanitized result. `execution_archive_result_sha256` retains the pre-sanitization execution digest.
5. Digests embedded **inside historical results/logs/plans** retain their original execution meaning.
   For example, a historical script/result/protocol digest may identify the original bytes before
   local-path removal, while upstream full-file and logical-array digests continue to identify the
   same public data. Use `SOURCE_MAP.json` for the original-to-public file mapping; do not assume
   every nested historical digest is the checksum of a present-day derivative file.

Redacted local paths use distinct deterministic tokens, so different path keys remain distinct.
Public upstream dataset URLs, revisions, input digests, numeric results, rules and tolerances are
preserved. The full historical source archive and private development history are not distributed.
