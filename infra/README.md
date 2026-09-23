# AWS Deployment

The pipeline is designed to run identically locally or on AWS — only the
storage/compute backend changes. This is a lightweight, low-cost path suitable
for a portfolio demo (not a production HA setup).

## Components

| Local dev              | AWS equivalent                                   |
|-------------------------|---------------------------------------------------|
| `data/raw`, `data/processed` | S3 bucket (`S3_BUCKET_NAME` in `.env`)       |
| `rag/index/*.faiss`     | S3-backed FAISS index (download to /tmp on cold start) |
| CLI scripts             | Lambda functions (ingestion + eval as scheduled jobs) or an ECS Fargate task for longer batch runs |
| Streamlit app           | ECS Fargate service or EC2 instance behind an ALB, or Streamlit Community Cloud for a free demo link |

## Suggested minimal setup

1. **S3 bucket**: store raw filings, processed JSON, and the FAISS index files.
   ```
   aws s3 mb s3://wealth-doc-intelligence
   ```

2. **Ingestion + indexing as a scheduled batch job**:
   - Package `ingestion/`, `models/`, and `rag/` into a container image.
   - Run as an ECS Fargate task (scheduled via EventBridge) rather than Lambda,
     since the HF models (BART, FinBERT) exceed Lambda's package size/memory
     limits comfortably handled by Fargate.
   - Task writes processed docs + FAISS index back to S3.

3. **Query API**:
   - A small FastAPI/Flask app wrapping `rag/retriever.answer_query`, deployed
     on the same Fargate service or a separate small ECS service.
   - API Gateway (HTTP API) in front if you want a stable public endpoint.

4. **Demo frontend**:
   - Easiest: deploy `app/streamlit_app.py` on Streamlit Community Cloud
     (free) pointed at your S3-hosted index via `boto3`, for a shareable
     portfolio link without managing AWS compute for the UI itself.
   - AWS-native alternative: containerize and run on ECS Fargate behind an ALB.

## IAM

Minimum policy for the ingestion/eval task role: `s3:GetObject`, `s3:PutObject`,
`s3:ListBucket` scoped to your bucket ARN. No other AWS permissions are needed
for this pipeline.

## Cost notes

- FAISS index + processed JSON for ~50 companies' 10-Ks is a few hundred MB —
  trivial S3 storage cost.
- Fargate task run on a schedule (e.g. weekly re-ingestion) costs cents per run
  if scoped to a short-lived batch job rather than an always-on service.
- Avoid SageMaker endpoints for this scale — a Fargate task loading HF models
  on demand is cheaper and simpler for a portfolio project.
