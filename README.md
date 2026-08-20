# LLM Cost Autopilot

A cost-aware LLM routing system that predicts the cheapest model likely to answer a prompt successfully, with optional quality verification, escalation, usage analytics, and a live React dashboard.

## Live Demo

**Frontend**

https://llm-cost-autopilot-dun.vercel.app

**Backend API**

https://llm-cost-autopilot-production-3ac9.up.railway.app

**Swagger**

https://llm-cost-autopilot-production-3ac9.up.railway.app/docs

---

## Why I Built This

LLM applications commonly send every request to the strongest available model.

That is simple, but it can be unnecessarily expensive.

A prompt such as:

> What is Python?

usually does not need the same model capability as:

> Design a globally distributed payment processing platform and discuss consistency, retries, failure recovery, race conditions, and disaster recovery.

LLM Cost Autopilot tries to solve this as a routing problem:

> Choose the cheapest model predicted to satisfy the request while maintaining acceptable answer quality.

The core objective is:

```text
minimize cost and latency

subject to

acceptable response quality

Features
ML-based LLM routing
Three model capability tiers
Calibrated per-tier success prediction
Cost-aware routing thresholds
Economy and Balanced generation modes
Selective LLM-based quality verification
Automatic model escalation
Token, latency, and cost accounting
SQLite request logging
Usage analytics
Estimated Tier-3 token-equivalent savings
Request history
FastAPI REST API
React dashboard and prompt playground
Docker and Docker Compose support
Railway backend deployment
Vercel frontend deployment
Automated

                         USER
                          │
                          ▼
                 React Dashboard
                     Vercel
                          │
                          ▼
                    FastAPI API
                     Railway
                          │
                          ▼
                 GenerationService
                          │
                          ▼
                     ML Router
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
          Tier 1       Tier 2       Tier 3
          cheapest      medium      strongest
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                     Gemini API
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
              Economy           Balanced
                 │                 │
              Return       Selective Verify
                                   │
                              ┌────┴────┐
                              ▼         ▼
                            PASS       FAIL
                              │         │
                            Return   Escalate
                                        │
                                        ▼
                                  Higher Tier
                          │
                          ▼
                   Usage Logging
                          │
                          ▼
                       SQLite
                          │
                          ▼
                     Analytics


Model Tiers

The current deployment uses three Gemini model tiers.

Tier	Model	Purpose
Tier 1	gemini-3.1-flash-lite	Cheapest model for simpler prompts
Tier 2	gemini-3.5-flash-lite	Mid-tier model for more demanding requests
Tier 3	gemini-3.7-flash	Strongest fallback / verifier model

The ML router operates on abstract tiers rather than directly depending on model IDs.

That allows the underlying provider models to be changed independently of the routing logic.

ML Routing Approach
Initial approach

The first router treated the problem as prompt-complexity classification:

simple
moderate
complex

using handcrafted prompt features and TF-IDF.

Although the early classifier performed well on a small manually created dataset, the dataset was too small and template-heavy to trust.

The project was therefore redesigned around a more useful question:

What is the probability that each model tier will successfully answer this prompt?

Training Data

The routing experiments used the SPROUT LLM-routing dataset.

The dataset contains model-response quality scores that were used as capability proxies for three model tiers.

A quality score of:

>= 0.80

was treated as a successful response.

The project used separate train, validation, and untouched test splits.

Per-Tier Success Predictors

Instead of training one multiclass model, the final router trains three independent binary predictors:

P(Tier 1 succeeds | prompt)


P(Tier 2 succeeds | prompt)


P(Tier 3 succeeds | prompt)

Features combine:

TF-IDF word and bigram features
prompt length
sentence count
question count
reasoning-related keywords
technical keywords
basic code indicators

Each predictor uses calibrated Logistic Regression.

Routing Policy

The application chooses the cheapest tier whose predicted success probability passes its routing threshold.

The frozen routing policy is:

Tier 1 score >= 0.80
    → Tier 1


else Tier 2 score >= 0.70
    → Tier 2


otherwise
    → Tier 3

The thresholds were selected using validation data.

The final test set was kept untouched during threshold tuning.

Router Test Results

On the untouched test split, the frozen policy achieved approximately:

Metric	Result
Overall success	78.85%
Success on solvable prompts	88.74%
Under-routing rate	4.93%
Average selected tier	2.25
Tier 1 usage	18.07%
Tier 2 usage	38.89%
Tier 3 usage	43.05%

Validation and test behavior were close, indicating that the routing policy generalized reasonably well rather than only fitting the validation split.

Generation Modes
Economy

Economy mode trusts the ML router.

Prompt
  ↓
ML Router
  ↓
Selected Model
  ↓
Return

It does not perform additional runtime verification.

Use this mode when minimizing cost and latency is the main priority.

Balanced

Balanced mode adds a selective safety layer.

Prompt
  ↓
ML Router
  ↓
Selected Model
  ↓
Verification Policy
  ↓
Verify only when needed
  ↓
PASS → Return


FAIL → Escalate

High-confidence decisions can skip verification.

Lower-confidence responses can be evaluated by a stronger model and escalated when necessary.

Selective Verification

Verifying every cheap-model response can undermine the cost savings achieved by routing.

The project therefore uses an auto-accept threshold:

routing confidence >= 0.90
→ skip verification

For lower-confidence non-Tier-3 requests:

generate
→ verify
→ escalate if verification fails

Tier 3 is returned directly because there is no stronger configured tier available for escalation.

Cost Accounting

Every generation attempt tracks:

input tokens
output tokens
thinking tokens
generation latency
verification latency
generation cost
verification cost
total attempt cost

For escalated requests, costs from all attempts are accumulated.

The project includes automated accounting checks verifying that:

sum(attempt costs)
==
reported request cost
Initial Engineering Benchmark

Three strategies were compared:

Always use Tier 3
ML routing only
Full Autopilot with selective verification

A small three-prompt benchmark containing simple, moderate, and complex prompts produced:

Strategy	Total Estimated Cost	Avg Quality
Always Tier 3	$0.02915325	1.000
ML Routing Only	$0.00777105	0.983
Full Autopilot	$0.01179755	0.983

All tested responses passed the benchmark evaluator.

In this small engineering benchmark, ML-only routing reduced estimated generation cost by approximately:

73.3%

relative to always using Tier 3.

Full Autopilot reduced estimated cost by approximately:

59.5%

The verifier introduced additional cost and latency without improving the final result on these three prompts because no escalation was required.

Important

This is a small application-level engineering benchmark, not a statistically rigorous universal performance claim.

Different prompts, model versions, pricing, and evaluator choices may produce different results.

Analytics

Every successful API generation is logged to SQLite without storing the user's prompt or generated response text.

Stored metadata includes:

request ID
timestamp
generation mode
initial tier
final tier
model
routing scores
verification state
escalation state
attempt count
token usage
estimated cost
latency

The API exposes:

GET /analytics/summary

for aggregate usage information.

GET /analytics/savings

for cost comparison against a Tier-3 token-equivalent baseline.

GET /analytics/requests

for recent request history.

Savings Metric

The dashboard calculates:

actual estimated cost


vs


same generation-token workload
priced using Tier-3 rates

This is reported as:

Tier-3 token-equivalent savings

It is intentionally not described as exact money saved because different models may generate different numbers of tokens for the same prompt.

API
Health
GET /health
Generate
POST /generate

Example:

{
  "prompt": "Explain authentication and authorization with examples.",
  "mode": "balanced"
}

Example response structure:

{
  "request_id": "...",
  "text": "...",
  "mode": "balanced",
  "initial_tier": "tier_2",
  "final_tier": "tier_2",
  "model_id": "gemini-3.5-flash-lite",
  "routing_scores": {
    "tier_1": 0.67,
    "tier_2": 0.79,
    "tier_3": 0.92
  },
  "verification_performed": true,
  "verification_passed": true,
  "verification_score": 1.0,
  "escalated": false,
  "total_estimated_cost_usd": 0.00694,
  "total_latency_ms": 21153
}
Analytics Summary
GET /analytics/summary
Savings
GET /analytics/savings
Recent Requests
GET /analytics/requests?limit=10
Tech Stack
Machine Learning
Python
scikit-learn
Logistic Regression
Calibrated classifiers
TF-IDF
joblib
SPROUT routing dataset
AI
Gemini API
LLM-as-a-judge verification
Dynamic model routing
Selective quality verification
Automatic escalation
Backend
FastAPI
Pydantic
Pydantic Settings
Uvicorn
SQLite
Frontend
React
Vite
React Markdown
Fetch API
Plain CSS
Infrastructure
Docker
Docker Compose
Railway
Vercel
Git / GitHub
Testing
pytest
FastAPI TestClient
mocked provider failures
temporary SQLite databases
Automated Tests

The project currently includes 19 automated tests covering:

router behavior
routing score structure
invalid router input
selective verification policy
SQLite request logging
analytics summaries
savings calculations
empty database behavior
FastAPI health endpoint
successful generation API responses
request validation
provider quota errors
provider availability errors
generic provider failures

Run:

python -m pytest -v
Project Structure
llm-cost-autopilot/
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   └── schemas/
│   │
│   ├── classifier/
│   ├── core/
│   ├── providers/
│   ├── services/
│   └── main.py
│
├── artifacts/
│   └── router/
│       ├── feature_transformer.joblib
│       ├── tier_1_predictor.joblib
│       ├── tier_2_predictor.joblib
│       ├── tier_3_predictor.joblib
│       └── router_config.json
│
├── frontend/
│
├── scripts/
│
├── tests/
│
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .env.example
└── README.md
Running Locally

Clone the repository:

git clone <repository-url>


cd llm-cost-autopilot

Create a virtual environment:

python3.11 -m venv venv


source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create:

.env

from:

.env.example

and configure:

GEMINI_API_KEY=your_api_key
APP_ENV=development
DATABASE_PATH=data/autopilot.db
VERIFICATION_AUTO_ACCEPT_SCORE=0.90

Run the backend:

python -m uvicorn app.main:app --reload

Swagger:

http://localhost:8000/docs
Running the Frontend
cd frontend


npm install

Create:

frontend/.env

with:

VITE_API_BASE_URL=http://localhost:8000

Start:

npm run dev

Open:

http://localhost:5173
Docker

Build and run using Docker Compose:

docker compose up -d --build

Check:

docker compose ps

View logs:

docker compose logs -f

Stop:

docker compose down

The local SQLite database can be stored in a persistent Docker volume.

Deployment
Backend

The FastAPI backend is deployed on Railway using the project's Dockerfile.

Runtime secrets are injected through Railway environment variables.

SQLite is stored on a persistent Railway volume mounted at:

/app/data
Frontend

The React/Vite frontend is deployed on Vercel.

The frontend communicates with the Railway API using:

VITE_API_BASE_URL
Limitations

This project intentionally remains scoped as a learning and portfolio project.

Current limitations include:

small end-to-end benchmark sample
LLM evaluator may introduce evaluator bias
routing was trained using proxy model capability data rather than Gemini-specific training labels
TF-IDF representation can be sensitive to wording
SQLite is appropriate for the current single-instance project but not ideal for large horizontally scaled deployments
savings analytics use a token-equivalent baseline rather than replaying every request through Tier 3
model pricing and capabilities may change over time
Possible Future Improvements

If the system were expanded beyond a portfolio project:

collect Gemini-specific routing feedback
replace TF-IDF with semantic embeddings or a small transformer router
optimize routing based on both quality and latency
learn verification thresholds from production feedback
migrate analytics storage to PostgreSQL
run larger independent quality evaluations
support additional LLM providers
add provider-level fallback routing
What I Learned

This project explored the full path from an ML experiment to a deployed AI system:

dataset analysis
        ↓
feature engineering
        ↓
ML model training
        ↓
probability calibration
        ↓
routing optimization
        ↓
untouched test evaluation
        ↓
artifact serialization
        ↓
service architecture
        ↓
LLM integration
        ↓
quality verification
        ↓
cost accounting
        ↓
FastAPI
        ↓
analytics
        ↓
automated testing
        ↓
Docker
        ↓
cloud deployment
        ↓
React dashboard

The biggest lesson was that optimizing an LLM application is not simply about selecting the smallest model.

Routing, verification, output length, latency, escalation frequency, and quality requirements all interact with total system cost.



## After adding it


Run:


```bash
git add README.md
git commit -m "Add project documentation"
git push

Then GitHub will finally have a proper landing page instead of immediately dropping a recruiter into source files.