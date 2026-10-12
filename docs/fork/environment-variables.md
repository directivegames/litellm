# Fork environment variables

Settings read by this tree that the LiteLLM docs checkout does not mention. The documentation check treats a name as documented when it appears on this page.

| Name | Description |
|------|-------------|
| PROXY_VIDEO_COST_POLLING_INTERVAL | Seconds between polls that bill a video job once it finishes. The caller's spend lags the finished job by at most one interval, plus up to 10 seconds of stagger. Default is 60. Values below 1 are raised to 1. |
| AGENT365_API_BASE | Base URL of the Microsoft Agent 365 tool-evaluation endpoint for the `agent_365` guardrail, used when `api_base` is unset. Default is `https://agent365.svc.cloud.microsoft`. |
| AGENT365_RESOURCE_APP_ID | Application id the `agent_365` guardrail mints its On-Behalf-Of token for, used when `resource_app_id` is unset. Default is the production Agent Tools app `ea9ffc3e-8a23-4a7d-836d-234d7c7565c1`. Test and PreProd use a different id. |
