const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;


export async function getAnalyticsSummary() {
  const response = await fetch(
    `${API_BASE_URL}/analytics/summary`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load analytics summary."
    );
  }

  return response.json();
}


export async function getSavingsSummary() {
  const response = await fetch(
    `${API_BASE_URL}/analytics/savings`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load savings summary."
    );
  }

  return response.json();
}


export async function getRecentRequests(
  limit = 5
) {
  const response = await fetch(
    `${API_BASE_URL}/analytics/requests?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load request history."
    );
  }

  return response.json();
}

export async function generateResponse(
  prompt,
  mode
) {
  const response = await fetch(
    `${API_BASE_URL}/generate`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        prompt,
        mode,
      }),
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data?.detail?.message ||
      data?.detail ||
      "Generation failed."
    );
  }

  return data;
}