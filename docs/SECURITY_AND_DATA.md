# Security, data provenance and deployment boundaries

## Not for production network operations

The API offers descriptive QoS estimates for an academic demonstration; it does not provide validated network performance assurance, SLA verification, or automated operator changes.

## Data sharing

No original operator CSV/XLSX measurements, personally identifying metadata, Flutter release binaries, credentials, connection strings or serialized model weights are intended in this public repository. Check licensing and dataset provenance before publishing source or derived measurements.

## Model file trust

`joblib` uses Python pickle mechanisms. Loading an untrusted artifact can execute arbitrary code. Use only files trained locally from trusted code and authorized data. The 3G route intentionally returns HTTP 503 when its trusted local model files are missing.

## Sensitive outputs

Do not commit generated `evaluation_3g.json`, raw dataset previews, notebook executions containing private fields, or debugging logs. Tests use fabricated inputs; they are not copied from field measurements.

## API scope

FastAPI demo routes are unauthenticated and designed for local development. To expose them publicly, add authentication, rate limiting, input validation, logging safeguards, secret management and deployment hardening.

## Responsible interpretation

3G classes are functions of input KPI thresholds; performance on those engineered labels does not establish independent QoS truth. 4G results are a weighted rubric rather than an externally calibrated classifier.
