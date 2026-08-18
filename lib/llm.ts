// Groq chat helper used by the runtime pipeline (generation + verification).
// Uses JSON Object Mode (response_format json_object). Because Groq does not
// guarantee strict schema adherence, every call validates its parsed output
// against a per-call validator and retries once on a schema/parse miss.

// Runtime model is configurable via GROQ_MODEL (server-side env); falls back to
// a default so the app still works when the env var is absent.
export const GROQ_MODEL = process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile';
const GROQ_BASE_URL = 'https://api.groq.com/openai/v1/chat/completions';

/** Call Groq once and return the raw assistant message content string. */
async function groqChat(system: string, user: string): Promise<string> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY is not set');
  const res = await fetch(GROQ_BASE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      response_format: { type: 'json_object' },
    }),
  });
  if (!res.ok) throw new Error(`LLM failed: ${res.status} ${await res.text()}`);
  const json = await res.json();
  return json.choices[0].message.content as string;
}

/**
 * Structured JSON request with schema validation + a single retry.
 * `validate` must be a type guard confirming the expected shape.
 */
export async function chatJSON<T>(
  system: string,
  user: string,
  validate: (data: unknown) => data is T,
  retries = 1,
): Promise<T> {
  let lastErr: Error | null = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const content = await groqChat(system, user);
      const parsed = JSON.parse(content) as unknown;
      if (validate(parsed)) return parsed;
      lastErr = new Error('LLM output failed schema validation');
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err));
      if (lastErr.message === 'LLM output failed schema validation' && attempt < retries) {
        continue; // schema miss: retry
      }
      // Network/API errors are not retried here; surface them.
      throw lastErr;
    }
  }
  throw lastErr;
}

