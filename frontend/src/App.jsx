import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  getAnalyticsSummary,
  getSavingsSummary,
  getRecentRequests,
  generateResponse,
} from "./api";

import "./App.css";


function App() {
  const [summary, setSummary] =
    useState(null);

  const [savings, setSavings] =
    useState(null);

  const [requests, setRequests] =
    useState([]);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [prompt, setPrompt] =
    useState("");
  
  const [mode, setMode] =
    useState("economy");
  
  const [generation, setGeneration] =
    useState(null);
  
  const [generating, setGenerating] =
    useState(false);
  
  const [generationError, setGenerationError] =
    useState("");


  useEffect(() => {
    loadDashboard();
  }, []);


  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");

      const [
        summaryData,
        savingsData,
        requestData,
      ] = await Promise.all([
        getAnalyticsSummary(),
        getSavingsSummary(),
        getRecentRequests(5),
      ]);

      setSummary(summaryData);
      setSavings(savingsData);
      setRequests(requestData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }


  if (loading) {
    return (
      <main className="page">
        <p>Loading dashboard...</p>
      </main>
    );
  }


  if (error) {
    return (
      <main className="page">
        <h1>LLM Cost Autopilot</h1>

        <p className="error">
          {error}
        </p>

        <button onClick={loadDashboard}>
          Retry
        </button>
      </main>
    );
  }

  async function handleGenerate(event) {
    event.preventDefault();
  
    if (!prompt.trim()) {
      return;
    }
  
    try {
      setGenerating(true);
      setGenerationError("");
      setGeneration(null);
  
      const result =
        await generateResponse(
          prompt.trim(),
          mode
        );
  
      setGeneration(result);
  
      // Refresh analytics because
      // /generate creates a new usage record.
      await loadDashboard();
  
    } catch (err) {
      setGenerationError(
        err.message
      );
    } finally {
      setGenerating(false);
    }
  }


  return (
    <main className="page">

      <header className="header">
        <div>
          <p className="eyebrow">
            AI ROUTING DASHBOARD
          </p>

          <h1>
            LLM Cost Autopilot
          </h1>

          <p className="subtitle">
            Route requests to the cheapest
            capable model while tracking cost,
            latency and quality safeguards.
          </p>
        </div>

        <button
          className="refresh"
          onClick={loadDashboard}
        >
          Refresh
        </button>
      </header>


      <section className="cards">

        <MetricCard
          label="Total Requests"
          value={summary.total_requests}
        />

        <MetricCard
          label="Total Cost"
          value={
            `$${summary.total_cost_usd.toFixed(4)}`
          }
        />

        <MetricCard
          label="Estimated Savings"
          value={
            `${savings.estimated_savings_percentage.toFixed(1)}%`
          }
        />

        <MetricCard
          label="Average Latency"
          value={
            `${(
              summary.average_latency_ms /
              1000
            ).toFixed(2)}s`
          }
        />

      </section>


      <section className="panel">

        <div className="panelHeader">
          <div>
            <h2>Model Usage</h2>
            <p>
              Distribution of final model tiers.
            </p>
          </div>
        </div>


        <div className="tiers">

          {Object.entries(
            summary.tier_usage
          ).map(([tier, data]) => (

            <div
              className="tier"
              key={tier}
            >

              <div className="tierTop">

                <span>
                  {formatTier(tier)}
                </span>

                <strong>
                  {data.percentage.toFixed(1)}%
                </strong>

              </div>


              <div className="bar">

                <div
                  className="barFill"
                  style={{
                    width:
                      `${data.percentage}%`,
                  }}
                />

              </div>


              <small>
                {data.count} requests
              </small>

            </div>

          ))}

        </div>

      </section>


      <section className="panel">

        <div className="panelHeader">

          <div>
            <h2>Recent Requests</h2>

            <p>
              Latest routing decisions recorded
              by the API.
            </p>
          </div>

        </div>


        <section className="playground">

  <div className="playgroundIntro">

    <p className="eyebrow">
      LIVE PLAYGROUND
    </p>

    <h2>
      Route a prompt
    </h2>

    <p>
      Enter a prompt and let the ML router
      choose the most cost-efficient model.
    </p>

  </div>


  <form
    className="promptForm"
    onSubmit={handleGenerate}
  >

    <textarea
      value={prompt}
      onChange={(event) =>
        setPrompt(
          event.target.value
        )
      }
      placeholder={
        "Ask something...\n\nFor example: Explain OAuth with an example."
      }
      rows={7}
    />


    <div className="formBottom">

      <div className="modeSelector">

        <button
          type="button"
          className={
            mode === "economy"
              ? "modeButton active"
              : "modeButton"
          }
          onClick={() =>
            setMode("economy")
          }
        >
          Economy
        </button>


        <button
          type="button"
          className={
            mode === "balanced"
              ? "modeButton active"
              : "modeButton"
          }
          onClick={() =>
            setMode("balanced")
          }
        >
          Balanced
        </button>

      </div>


      <button
        className="generateButton"
        type="submit"
        disabled={
          generating ||
          !prompt.trim()
        }
      >
        {
          generating
            ? "Generating..."
            : "Generate"
        }
      </button>

    </div>

  </form>


  {generationError && (

    <p className="generationError">
      {generationError}
    </p>

  )}


  {generation && (

    <div className="generationResult">

      <div className="resultHeader">

        <div>
          <p className="eyebrow">
            RESPONSE
          </p>

          <h3>
            {generation.model_id}
          </h3>
        </div>


        <span className="tierBadge">
          {
            formatTier(
              generation.final_tier
            )
          }
        </span>

      </div>


      <div className="responseText">
      <ReactMarkdown
  remarkPlugins={[remarkGfm]}
>
  {generation.text}
</ReactMarkdown>
</div>


      <div className="resultMetrics">

        <ResultMetric
          label="Mode"
          value={generation.mode}
        />

        <ResultMetric
          label="Initial Tier"
          value={
            formatTier(
              generation.initial_tier
            )
          }
        />

        <ResultMetric
          label="Final Tier"
          value={
            formatTier(
              generation.final_tier
            )
          }
        />

        <ResultMetric
          label="Cost"
          value={
            `$${generation
              .total_estimated_cost_usd
              .toFixed(6)}`
          }
        />

        <ResultMetric
          label="Latency"
          value={
            `${(
              generation
                .total_latency_ms /
              1000
            ).toFixed(2)}s`
          }
        />

        <ResultMetric
          label="Verified"
          value={
            generation
              .verification_performed
              ? "Yes"
              : "No"
          }
        />

        <ResultMetric
          label="Escalated"
          value={
            generation.escalated
              ? "Yes"
              : "No"
          }
        />

      </div>


      <div className="routingScores">

        <h4>
          Router confidence
        </h4>

        {Object.entries(
          generation.routing_scores
        ).map(
          ([tier, score]) => (

            <div
              className="scoreRow"
              key={tier}
            >

              <span>
                {formatTier(tier)}
              </span>

              <div className="scoreBar">

                <div
                  className="scoreFill"
                  style={{
                    width:
                      `${score * 100}%`,
                  }}
                />

              </div>

              <strong>
                {
                  (
                    score * 100
                  ).toFixed(1)
                }%
              </strong>

            </div>

          )
        )}

      </div>

    </div>

  )}

</section>


        {requests.length === 0 ? (

          <p className="empty">
            No requests recorded yet.
          </p>

        ) : (

          <div className="tableWrapper">

            <table>

              <thead>
                <tr>
                  <th>Mode</th>
                  <th>Tier</th>
                  <th>Model</th>
                  <th>Cost</th>
                  <th>Latency</th>
                  <th>Verified</th>
                </tr>
              </thead>


              <tbody>

                {requests.map(
                  (request) => (

                    <tr
                      key={
                        request.request_id
                      }
                    >

                      <td>
                        {request.mode}
                      </td>

                      <td>
                        {formatTier(
                          request.final_tier
                        )}
                      </td>

                      <td>
                        {request.model_id}
                      </td>

                      <td>
                        $
                        {request
                          .total_cost_usd
                          .toFixed(5)}
                      </td>

                      <td>
                        {(
                          request
                            .total_latency_ms /
                          1000
                        ).toFixed(2)}
                        s
                      </td>

                      <td>
                        {
                          request
                            .verification_performed
                            ? "Yes"
                            : "No"
                        }
                      </td>

                    </tr>

                  )
                )}

              </tbody>

            </table>

          </div>

        )}

      </section>

    </main>
  );
}


function MetricCard({
  label,
  value,
}) {
  return (
    <article className="card">

      <p>
        {label}
      </p>

      <strong>
        {value}
      </strong>

    </article>
  );
}

function ResultMetric({
  label,
  value,
}) {
  return (
    <div className="resultMetric">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}


function formatTier(tier) {
  return tier
    .replace("_", " ")
    .replace(
      /\b\w/g,
      (letter) =>
        letter.toUpperCase()
    );
}


export default App;