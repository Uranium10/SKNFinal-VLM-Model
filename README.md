# BiddingFlow quotation extraction worker

This repository is an isolated RunPod Serverless queue worker. It performs only
multimodal inference and never connects directly to ERPNext or PostgreSQL.

## Request contract

The BiddingFlow backend owns the prompt registry. It sends a versioned prompt
and its SHA-256 hash with one or more short-lived document URLs. No production
prompt is stored in this repository. Base64 input is supported for small
console tests, but URLs are preferred because RunPod `/run` requests have a
payload limit.

```json
{
  "input": {
    "request_id": "quotation-PUR-RFQ-2026-00001-001",
    "documents": [
      {
        "url": "https://biddingflow.example/internal/files/token",
        "filename": "quotation.pdf"
      }
    ],
    "system_prompt": "...",
    "user_prompt": "...",
    "prompt_version": "quotation_json_v1",
    "prompt_sha256": "...",
    "max_new_tokens": 512
  },
  "webhook": "https://biddingflow.example/api/webhooks/runpod/quotation"
}
```

All four prompt fields (`system_prompt`, `user_prompt`, `prompt_version`, and
`prompt_sha256`) are required for extraction requests. The Worker recalculates
the hash before inference and rejects mismatches.

The Worker validates the model JSON before returning it. Set
`INCLUDE_RAW_MODEL_OUTPUT=true` only for temporary diagnosis because raw output
may contain quotation data.

## RunPod GitHub deployment

1. RunPod **Settings > Connections > GitHub > Connect** and grant access only
   to the backend repository.
2. Choose **Serverless > New Endpoint > Import Git Repository**.
3. Select branch `main` and the root `Dockerfile`.
4. Select a queue endpoint and initially configure:
   - Active workers: `0`
   - Max workers: `1`
   - GPUs per worker: `1`
   - GPU memory: `24 GB` or more
   - Idle timeout: `60` seconds
   - Execution timeout: `300` seconds
   - FlashBoot: enabled
5. Set the cached model to `Qwen/Qwen3.5-9B`.

The Docker build context and Dockerfile are both at the repository root.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BASE_MODEL_ID` | `Qwen/Qwen3.5-9B` | Hugging Face base model |
| `BASE_MODEL_REVISION` | empty | Optional pinned base revision |
| `ADAPTER_MODEL_ID` | `lyc9872/qwen_3.5_9b_peft` | LoRA adapter |
| `ADAPTER_MODEL_REVISION` | pinned evaluated revision | Adapter revision |
| `VISION_MIN_PIXELS` | `200704` | Evaluation-compatible minimum pixels |
| `VISION_MAX_PIXELS` | `802816` | Evaluation-compatible maximum pixels |
| `MAX_NEW_TOKENS` | `512` | Default output token limit |
| `MAX_NEW_TOKENS_CAP` | `1024` | Per-request hard limit |
| `ALLOWED_DOCUMENT_HOSTS` | empty | Comma-separated download host allowlist |
| `ALLOW_HTTP_DOCUMENT_URLS` | `false` | Allow plain HTTP only for controlled tests |
| `MAX_DOCUMENT_BYTES` | `26214400` | Download/base64 size limit |
| `MAX_PDF_PAGES` | `8` | PDF page limit |
| `INCLUDE_RAW_MODEL_OUTPUT` | `false` | Include raw generated text in response |

The Hugging Face repositories are currently public. If that changes, add
`HF_TOKEN` through RunPod Secrets rather than committing it.

## Contract tests

These tests do not load the GPU model:

```bash
pytest tests -q
```

`test_input.json` uses the health-check request so container/build checks do
not spend GPU time or require a quotation fixture.
