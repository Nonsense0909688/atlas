"""Built-in Atlas test cases — the benchmark suite.

Each ``TestCase`` (see ``models.py``) is one scenario: a prompt, the tools
available, the ground-truth tool calls a perfect model should emit, canned
mock outputs, and metadata (category, difficulty, rubric).

Coverage (32 cases):
    single_tool         — basic single calls (weather, math, search)
    complex_args        — many params, arrays (flights, email, calendar)
    multi_tool          — sequential multi-tool chains
    tool_selection      — disambiguation between similar tools
    abstention          — no-tool cases (greetings, opinions)
    required_args       — missing-required-parameter handling
    optional_args       — optional params: omit vs specify
    type_safety         — wrong types (string for number, etc.)
    hallucination       — inventing tools/params
    parallel_calls      — independent calls in one turn
    dependency_chain    — tool B depends on tool A's output
    conversation        — multi-turn call sequences
    instruction_conflict— "don't use tools" / "must use tools"
    edge                — ambiguous prompts

Difficulties: easy (single, obvious) < medium (multi-param, disambiguate)
< hard (multi-tool, missing info, traps) < expert (chains, parallel+seq).

To add a case: append a ``TestCase`` below. Reuse the ``_X_tool()``
factories so tool schemas stay consistent across the suite.
"""

from __future__ import annotations

from atlas.models import ExpectedToolCall, TestCase, ToolDefinition


# ── Tool factories ────────────────────────────────────────────────────
# One factory per tool so every case shares the exact same JSON Schema.
# Parametrized once here, validated everywhere — change it in one place.

def _weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string"},
            },
            "required": ["city"],
        },
    )


def _calculator_tool() -> ToolDefinition:
    return ToolDefinition(
        name="calculator",
        description="Evaluate a math expression",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )


def _search_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_web",
        description="Search the web for a query",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["query"],
        },
    )


def _book_tool() -> ToolDefinition:
    return ToolDefinition(
        name="book_flight",
        description="Book a flight",
        parameters={
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
                "passengers": {"type": "number"},
            },
            "required": ["origin", "destination", "date"],
        },
    )


def _send_email_tool() -> ToolDefinition:
    return ToolDefinition(
        name="send_email",
        description="Send an email",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    )


def _create_event_tool() -> ToolDefinition:
    return ToolDefinition(
        name="create_calendar_event",
        description="Create a calendar event",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string"},
                "end": {"type": "string"},
                "attendees": {"type": "array"},
            },
            "required": ["title", "start", "end"],
        },
    )


def _file_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file from disk",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "limit": {"type": "number"},
            },
            "required": ["path"],
        },
    )


def _db_tool() -> ToolDefinition:
    return ToolDefinition(
        name="query_database",
        description="Run a SQL query",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number"},
            },
            "required": ["query"],
        },
    )


# ── The suite ─────────────────────────────────────────────────────────

def get_builtin_cases() -> list[TestCase]:
    """Return all built-in test cases (fresh objects on every call)."""
    return [
        # ── Single tool ────────────────────────────────────────────────
        TestCase(
            id="weather-single",
            prompt="What's the weather in Tokyo?",
            tools=[_weather_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {"city": "Tokyo"})],
            mock_outputs={"get_weather": {"city": "Tokyo", "temp_c": 22, "condition": "clear"}},
            category="single_tool",
            difficulty="easy",
            rubric=["must call get_weather with city=Tokyo"],
        ),
        TestCase(
            id="weather-with-units",
            prompt="What's the weather in Paris in Fahrenheit?",
            tools=[_weather_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {"city": "Paris", "units": "fahrenheit"})],
            mock_outputs={"get_weather": {"city": "Paris", "temp_f": 72}},
            category="single_tool",
            difficulty="easy",
            rubric=["must include units, case-insensitive ok"],
        ),
        TestCase(
            id="math-single",
            prompt="What is 42 * 17?",
            tools=[_calculator_tool()],
            expected_calls=[ExpectedToolCall("calculator", {"expression": "42 * 17"})],
            mock_outputs={"calculator": {"result": 714}},
            category="single_tool",
            difficulty="easy",
            rubric=["must call calculator"],
        ),
        TestCase(
            id="search-single",
            prompt="Search for the latest news about quantum computing, top 5 results",
            tools=[_search_tool()],
            expected_calls=[ExpectedToolCall("search_web", {"query": "quantum computing", "limit": 5})],
            mock_outputs={"search_web": {"results": [{"title": "QC news"}]}},
            category="single_tool",
            difficulty="medium",
            rubric=["must call search_web with limit 5"],
        ),

        # ── Complex args ───────────────────────────────────────────────
        TestCase(
            id="flight-complex",
            prompt="Book a flight from NYC to LAX on 2026-12-01 for 2 passengers",
            tools=[_book_tool()],
            expected_calls=[
                ExpectedToolCall("book_flight", {"origin": "NYC", "destination": "LAX", "date": "2026-12-01", "passengers": 2})
            ],
            mock_outputs={"book_flight": {"confirmation": "ABC123"}},
            category="complex_args",
            difficulty="medium",
            rubric=["all 4 args correct"],
        ),
        TestCase(
            id="email-complex",
            prompt="Send an email to bob@example.com with subject 'Meeting tomorrow' and body 'Hi Bob, see you at 10am'",
            tools=[_send_email_tool()],
            expected_calls=[
                ExpectedToolCall("send_email", {"to": "bob@example.com", "subject": "Meeting tomorrow", "body": "Hi Bob, see you at 10am"})
            ],
            mock_outputs={"send_email": {"sent": True}},
            category="complex_args",
            difficulty="medium",
            rubric=["to/subject/body all correct"],
        ),
        TestCase(
            id="event-attendees",
            prompt="Create a meeting titled 'Sprint Planning' from 2026-11-06T09:00 to 2026-11-06T10:00 with Alice and Bob",
            tools=[_create_event_tool()],
            expected_calls=[
                ExpectedToolCall("create_calendar_event", {"title": "Sprint Planning", "start": "2026-11-06T09:00", "end": "2026-11-06T10:00", "attendees": ["Alice", "Bob"]})
            ],
            mock_outputs={"create_calendar_event": {"id": "evt_1"}},
            category="complex_args",
            difficulty="hard",
            rubric=["attendees array must match"],
        ),

        # ── Multi-tool ─────────────────────────────────────────────────
        TestCase(
            id="multi-weather-search",
            prompt="First check the weather in Berlin, then search the web for 'Berlin travel tips'",
            tools=[_weather_tool(), _search_tool()],
            expected_calls=[
                ExpectedToolCall("get_weather", {"city": "Berlin"}),
                ExpectedToolCall("search_web", {"query": "Berlin travel tips"}),
            ],
            mock_outputs={"get_weather": {"city": "Berlin", "temp_c": 15}, "search_web": {"results": []}},
            category="multi_tool",
            difficulty="medium",
            rubric=["must call both tools, weather first"],
        ),
        TestCase(
            id="multi-parallel",
            prompt="Get the weather in London and in New York at the same time",
            tools=[_weather_tool()],
            expected_calls=[
                ExpectedToolCall("get_weather", {"city": "London"}),
                ExpectedToolCall("get_weather", {"city": "New York"}),
            ],
            mock_outputs={"get_weather": {"ok": True}},
            category="multi_tool",
            difficulty="medium",
            rubric=["must call get_weather twice with correct cities"],
        ),
        TestCase(
            id="multi-three",
            prompt="Calculate 100 + 200, then check weather in Rome, then search for 'Rome weather history'",
            tools=[_calculator_tool(), _weather_tool(), _search_tool()],
            expected_calls=[
                ExpectedToolCall("calculator", {"expression": "100 + 200"}),
                ExpectedToolCall("get_weather", {"city": "Rome"}),
                ExpectedToolCall("search_web", {"query": "Rome weather history"}),
            ],
            mock_outputs={"calculator": {"result": 300}, "get_weather": {"city": "Rome"}, "search_web": {"results": []}},
            category="multi_tool",
            difficulty="hard",
            rubric=["all three tools, correct order"],
        ),

        # ── Tool selection ─────────────────────────────────────────────
        TestCase(
            id="disambiguate-calc-vs-search",
            prompt="What is 15% of 200?",
            tools=[_calculator_tool(), _search_tool(), _weather_tool()],
            expected_calls=[ExpectedToolCall("calculator", {"expression": "0.15 * 200"})],
            mock_outputs={"calculator": {"result": 30}},
            category="tool_selection",
            difficulty="medium",
            rubric=["must pick calculator, not search"],
        ),
        TestCase(
            id="disambiguate-weather-vs-search",
            prompt="Is it going to rain in Seattle this weekend? Search for forecasts if needed",
            tools=[_weather_tool(), _search_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {"city": "Seattle"})],
            mock_outputs={"get_weather": {"city": "Seattle", "rain_chance": 0.6}},
            category="tool_selection",
            difficulty="medium",
            rubric=["must pick get_weather first"],
        ),

        # ── Abstention ─────────────────────────────────────────────────
        TestCase(
            id="no-tool-greeting",
            prompt="Hello, how are you?",
            tools=[_weather_tool(), _calculator_tool()],
            expected_calls=[],
            mock_outputs={},
            category="abstention",
            difficulty="easy",
            rubric=["no tool calls expected"],
        ),
        TestCase(
            id="no-tool-opinion",
            prompt="What do you think about the meaning of life?",
            tools=[_search_tool()],
            expected_calls=[],
            mock_outputs={},
            category="abstention",
            difficulty="easy",
            rubric=["no tool calls expected"],
        ),

        # ── Edge ───────────────────────────────────────────────────────
        TestCase(
            id="weather-implicit-city",
            prompt="How's the weather?",
            tools=[_weather_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {})],
            mock_outputs={"get_weather": {"city": "unknown", "temp_c": 20}},
            category="edge",
            difficulty="hard",
            rubric=["should still attempt weather or ask clarifying (no-tool also scored leniently via overall)"],
        ),

        # ── Required args ──────────────────────────────────────────────
        TestCase(
            id="required-args-complete",
            prompt="Book a flight from SFO to JFK on 2026-12-15",
            tools=[_book_tool()],
            expected_calls=[ExpectedToolCall("book_flight", {"origin": "SFO", "destination": "JFK", "date": "2026-12-15"})],
            mock_outputs={"book_flight": {"confirmation": "XYZ789"}},
            category="required_args",
            difficulty="easy",
            rubric=["all required args present, optional passengers omitted is fine"],
        ),
        TestCase(
            id="required-args-missing-info",
            prompt="Book a flight to London",
            tools=[_book_tool()],
            expected_calls=[ExpectedToolCall("book_flight", {"destination": "London"})],
            mock_outputs={"book_flight": {"confirmation": "PARTIAL"}},
            category="required_args",
            difficulty="hard",
            rubric=["model should ask for missing origin/date, not hallucinate them"],
        ),

        # ── Optional args ──────────────────────────────────────────────
        TestCase(
            id="optional-args-omitted",
            prompt="Read the file at /tmp/report.txt",
            tools=[_file_tool()],
            expected_calls=[ExpectedToolCall("read_file", {"path": "/tmp/report.txt"})],
            mock_outputs={"read_file": {"content": "..."}},
            category="optional_args",
            difficulty="easy",
            rubric=["should not invent offset/limit values"],
        ),
        TestCase(
            id="optional-args-specified",
            prompt="Read the file at /tmp/data.csv starting from line 100, show 50 lines",
            tools=[_file_tool()],
            expected_calls=[ExpectedToolCall("read_file", {"path": "/tmp/data.csv", "offset": 100, "limit": 50})],
            mock_outputs={"read_file": {"content": "..."}},
            category="optional_args",
            difficulty="medium",
            rubric=["offset and limit must match exactly"],
        ),

        # ── Type safety ────────────────────────────────────────────────
        TestCase(
            id="type-safety-number",
            prompt="Search for 'AI news' with limit 10",
            tools=[_search_tool()],
            expected_calls=[ExpectedToolCall("search_web", {"query": "AI news", "limit": 10})],
            mock_outputs={"search_web": {"results": []}},
            category="type_safety",
            difficulty="medium",
            rubric=["limit must be number 10, not string '10'"],
        ),
        TestCase(
            id="type-safety-array",
            prompt="Create a calendar event 'Team Lunch' from 2026-11-07T12:00 to 2026-11-07T13:00 with attendees Charlie and Diana",
            tools=[_create_event_tool()],
            expected_calls=[
                ExpectedToolCall("create_calendar_event", {"title": "Team Lunch", "start": "2026-11-07T12:00", "end": "2026-11-07T13:00", "attendees": ["Charlie", "Diana"]})
            ],
            mock_outputs={"create_calendar_event": {"id": "evt_2"}},
            category="type_safety",
            difficulty="medium",
            rubric=["attendees must be array, not string"],
        ),

        # ── Hallucination ──────────────────────────────────────────────
        TestCase(
            id="hallucination-unknown-tool",
            prompt="What's the weather in Madrid?",
            tools=[_weather_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {"city": "Madrid"})],
            mock_outputs={"get_weather": {"city": "Madrid", "temp_c": 28}},
            category="hallucination",
            difficulty="easy",
            rubric=["must use get_weather, not invent get_forecast_api"],
        ),
        TestCase(
            id="hallucination-extra-param",
            prompt="What's the weather in Oslo?",
            tools=[_weather_tool()],
            expected_calls=[ExpectedToolCall("get_weather", {"city": "Oslo"})],
            mock_outputs={"get_weather": {"city": "Oslo", "temp_c": 5}},
            category="hallucination",
            difficulty="medium",
            rubric=["no extra params like humidity_threshold or forecast_days"],
        ),

        # ── Parallel calls ─────────────────────────────────────────────
        TestCase(
            id="parallel-weather-three",
            prompt="Get the weather in Tokyo, Sydney, and Paris simultaneously",
            tools=[_weather_tool()],
            expected_calls=[
                ExpectedToolCall("get_weather", {"city": "Tokyo"}),
                ExpectedToolCall("get_weather", {"city": "Sydney"}),
                ExpectedToolCall("get_weather", {"city": "Paris"}),
            ],
            mock_outputs={"get_weather": {"ok": True}},
            category="parallel_calls",
            difficulty="hard",
            rubric=["three parallel get_weather calls, order may vary but all cities present"],
        ),
        TestCase(
            id="parallel-mixed",
            prompt="Calculate 7 * 8 and search for 'multiplication tables' at the same time",
            tools=[_calculator_tool(), _search_tool()],
            expected_calls=[
                ExpectedToolCall("calculator", {"expression": "7 * 8"}),
                ExpectedToolCall("search_web", {"query": "multiplication tables"}),
            ],
            mock_outputs={"calculator": {"result": 56}, "search_web": {"results": []}},
            category="parallel_calls",
            difficulty="medium",
            rubric=["both tools called in one turn"],
        ),

        # ── Dependency chain ───────────────────────────────────────────
        TestCase(
            id="depchain-weather-search",
            prompt="Step 1: get weather for Chicago. Step 2: based on the result, search for activities.",
            tools=[_weather_tool(), _search_tool()],
            expected_calls=[
                ExpectedToolCall("get_weather", {"city": "Chicago"}),
            ],
            mock_outputs={"get_weather": {"city": "Chicago", "temp_c": 10, "condition": "rainy"}, "search_web": {"results": []}},
            category="dependency_chain",
            difficulty="hard",
            rubric=["must call get_weather first; search depends on its output (second turn)"],
        ),
        TestCase(
            id="depchain-calc-weather",
            prompt="Calculate the average of 10 and 20, then check the weather in the city you decide",
            tools=[_calculator_tool(), _weather_tool()],
            expected_calls=[
                ExpectedToolCall("calculator", {"expression": "(10 + 20) / 2"}),
            ],
            mock_outputs={"calculator": {"result": 15}, "get_weather": {"city": "Denver"}},
            category="dependency_chain",
            difficulty="expert",
            rubric=["calculator first, weather second depends on calc result"],
        ),

        # ── Conversation ───────────────────────────────────────────────
        TestCase(
            id="conversation-weather-then-calc",
            prompt="First: weather in Miami. Then: calculate the difference between 100 and the temperature.",
            tools=[_weather_tool(), _calculator_tool()],
            expected_calls=[
                ExpectedToolCall("get_weather", {"city": "Miami"}),
            ],
            mock_outputs={"get_weather": {"city": "Miami", "temp_c": 30}, "calculator": {"result": 70}},
            category="conversation",
            difficulty="expert",
            rubric=["multi-turn: weather call in turn 1, calculator in turn 2 uses weather result"],
        ),

        # ── Instruction conflict ───────────────────────────────────────
        TestCase(
            id="conflict-dont-call-tool",
            prompt="Don't use any tools. Just tell me what 5 * 5 equals.",
            tools=[_calculator_tool()],
            expected_calls=[],
            mock_outputs={},
            category="instruction_conflict",
            difficulty="hard",
            rubric=["user explicitly said don't use tools — must abstain even though calculator exists"],
        ),
        TestCase(
            id="conflict-use-tool",
            prompt="You must use the calculator tool to answer: what is 9 * 9?",
            tools=[_calculator_tool()],
            expected_calls=[ExpectedToolCall("calculator", {"expression": "9 * 9"})],
            mock_outputs={"calculator": {"result": 81}},
            category="instruction_conflict",
            difficulty="easy",
            rubric=["must use tool when explicitly instructed"],
        ),

        # ── Expert stretch ─────────────────────────────────────────────
        TestCase(
            id="expert-flight-email-chain",
            prompt="Book a flight from Boston to Denver on 2026-12-20 for 3 passengers, then send a confirmation email to alice@example.com",
            tools=[_book_tool(), _send_email_tool()],
            expected_calls=[
                ExpectedToolCall("book_flight", {"origin": "Boston", "destination": "Denver", "date": "2026-12-20", "passengers": 3}),
            ],
            mock_outputs={"book_flight": {"confirmation": "BOS-DEN-123"}, "send_email": {"sent": True}},
            category="dependency_chain",
            difficulty="expert",
            rubric=["book flight first, email depends on confirmation"],
        ),
        TestCase(
            id="expert-search-calc-weather",
            prompt="Search for 'best hiking trails Colorado', calculate 15 * 3, and check weather in Boulder — all in one go",
            tools=[_search_tool(), _calculator_tool(), _weather_tool()],
            expected_calls=[
                ExpectedToolCall("search_web", {"query": "best hiking trails Colorado"}),
                ExpectedToolCall("calculator", {"expression": "15 * 3"}),
                ExpectedToolCall("get_weather", {"city": "Boulder"}),
            ],
            mock_outputs={"search_web": {"results": []}, "calculator": {"result": 45}, "get_weather": {"city": "Boulder"}},
            category="parallel_calls",
            difficulty="expert",
            rubric=["all three tools in parallel"],
        ),
    ]
