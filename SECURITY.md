# Security policy

## Reporting a vulnerability

Please report security issues **privately**, not in a public issue: use GitHub's
[private vulnerability reporting](../../security/advisories/new) for this repository, or email
mosesroth@gmail.com. Include what you found, how to reproduce it, and its impact. You'll get an
acknowledgement within a week.

## Scope notes

- dev-titles runs locally and calls the LLM / ASR backends you configure; API keys come from your
  environment or the providers' own CLIs and are never written to the output directory.
- Inputs (subtitle files, transcripts, web pages fetched for term research) are treated as data:
  fetches are restricted to http(s), XML is parsed with `defusedxml`, and the test suite plus
  Bandit run on every push and pull request.
