"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, sys, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser



logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ───────────────────────────────────────────────────────
# Creates the ASGI server AgentCore Runtime will host. Exactly one instance
# per deployment.
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ─────────────────────────────────────────────────────────────
# Resource IDs collected in Part 1 of INSTRUCTIONS.md. Each also falls back to
# an environment variable of the same name, so the same code can be deployed
# to different environments (dev/staging/prod) without editing source —
# `agentcore deploy` can pass these in as runtime env vars instead of baking
# them into the file.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console

GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://customersupportgateway-5jqn8vtvo5.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp")
KB_ID       = os.environ.get("KB_ID", "6F3KSMNWOI")
REGION      = os.environ.get("REGION", "us-east-1")
MEMORY_ID   = os.environ.get("MEMORY_ID", "CustomerSupportMemory-tGQICv3D3G")


def _is_configured(value: str) -> bool:
    """A config value counts as 'set' once it's no longer the placeholder."""
    return bool(value) and not value.startswith("<")


# ── Model and Clients ─────────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id, region_name=REGION)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Namespace Helper ──────────────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string.

    Example output:
      { "SEMANTIC": "cs_agent/{actorId}/facts",
        "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }
    """
    if not _is_configured(memory_id):
        return {}
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
    except Exception as e:
        logger.warning(f"Could not load memory strategies: {e}")
        return {}
    return {s["type"]: s["namespaces"][0] for s in strategies}


# ── Memory Hook ────────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent.

    Retrieves relevant customer context (facts + preferences) before each
    response, and saves the completed turn afterwards so future sessions
    for the same customer can recall it.
    """

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    @staticmethod
    def _plain_text(message: dict):
        """Return the text of a message if it's a plain-text turn, else None.

        Tool-result messages have a `toolResult` block instead of `text`, so
        this doubles as the "not a tool result" check the hook needs.
        """
        content = message.get("content") or []
        if not content:
            return None
        first = content[0]
        return first.get("text") if isinstance(first, dict) and "text" in first else None

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        if not self.namespaces:
            return

        messages = event.agent.messages
        if not messages:
            return

        last = messages[-1]
        if last.get("role") != "user":
            return

        query = self._plain_text(last)
        if not query:
            return  # tool result, not a real user turn — skip

        memory_lines = []
        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed for {namespace}: {e}")
                continue

            for m in memories or []:
                text = (m.get("content") or {}).get("text") if isinstance(m, dict) else None
                if text:
                    memory_lines.append(f"[{strategy_type}] {text}")

        if memory_lines:
            context_block = "\n".join(memory_lines)
            last["content"][0]["text"] = f"Customer Context:\n{context_block}\n\n{query}"

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        if not _is_configured(self.memory_id):
            return

        messages = event.agent.messages
        customer_query = None
        agent_response = None

        for msg in reversed(messages):
            role = msg.get("role")
            text = self._plain_text(msg)
            if not text:
                continue
            if role == "assistant" and agent_response is None:
                agent_response = text
            elif role == "user" and customer_query is None:
                customer_query = text
            if customer_query and agent_response:
                break

        if not (customer_query and agent_response):
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
            )
        except Exception as e:
            logger.warning(f"Failed to save interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Knowledge Base Tool ───────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not _is_configured(KB_ID):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.error(f"Knowledge base retrieve failed: {e}")
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [
        r["content"]["text"]
        for r in results
        if r.get("content", {}).get("text")
    ]
    if not chunks:
        return "No relevant information found in the knowledge base."

    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ─────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

# Redeem points: 100 points = $1, minimum redemption chunk of 500 points,
# capped so redeemed value never exceeds 50% of the order total.
max_redeemable_value = order_total * 0.5
max_points_by_value = int(max_redeemable_value * 100)
redeemable_points = min(loyalty_points, max_points_by_value)
points_redeemed = (redeemable_points // 500) * 500
points_discount = points_redeemed / 100

subtotal_after_points = order_total - points_discount

tier_discount_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_discount_rate, 2)

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(order_total - final_total, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "tier": tier,
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_discount": points_discount,
    "tier_discount_rate": tier_discount_rate,
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            for event in response.get("stream", []):
                result = event.get("result")
                if not result:
                    continue

                content = result.get("content")
                if isinstance(content, list):
                    texts = [
                        c.get("text")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text" and c.get("text")
                    ]
                    if texts:
                        return "\n".join(texts)

                return json.dumps(result)

        return json.dumps({"error": "No result returned from the Code Interpreter."})

    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback calculation: {e}")
        # Fallback: tier discount only — no points redemption, no exact
        # sandbox execution. Good enough to keep the conversation moving
        # if the Code Interpreter resource is down or misconfigured.
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_rate = tier_rates.get(tier, 0.0)
        tier_discount = round(order_total * tier_discount_rate, 2)
        final_total = round(order_total - tier_discount, 2)
        return json.dumps({
            "tier": tier,
            "order_total": order_total,
            "tier_discount_rate": tier_discount_rate,
            "tier_discount": tier_discount,
            "final_total": final_total,
            "note": (
                "Code Interpreter unavailable — fallback applied tier discount "
                "only; points redemption was not calculated."
            ),
        })


# ── Agent Entrypoint ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, professional customer support assistant \
for an Amazon-style e-commerce store.

You can:
- Track orders and process refunds using the order/refund tools
- Answer product, policy, and loyalty-program questions using search_knowledge_base
- Calculate exact loyalty discounts using calculate_loyalty_discount
- Browse live web pages when a customer needs current information from a URL

Always ground factual claims about products, policies, or loyalty tiers in
search_knowledge_base rather than guessing. Use calculate_loyalty_discount for
any discount math instead of computing it yourself. Be concise and warm."""


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    user_input = payload.get("prompt", "")
    actor_id = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id") or str(uuid.uuid4())

    if not user_input:
        return "Please provide a prompt."

    memory_hook = MemoryHook(actor_id, session_id, memory_client, MEMORY_ID)
    agent_core_browser = AgentCoreBrowser(region=REGION)

    tools = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

    try:
        if _is_configured(GATEWAY_URL):
            mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
            with mcp_client:
                gateway_tools = mcp_client.list_tools_sync()
                tools.extend(gateway_tools)
                
                agent = Agent(
                    model=model,
                    tools=tools,
                    hooks=[memory_hook],
                    system_prompt=SYSTEM_PROMPT,
                )
                response = agent(user_input)
                return response.message["content"][0]["text"]
        else:
            logger.warning("GATEWAY_URL not configured — running without Gateway tools.")
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            response = agent(user_input)
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"Sorry, something went wrong while processing your request: {e}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        app.run()