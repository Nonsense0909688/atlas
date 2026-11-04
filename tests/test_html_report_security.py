from atlas.models import EvalResult, ScoreBreakdown
from atlas.runner import format_report_html


def test_html_report_escapes_model_and_case_content():
    result = EvalResult(
        case_id='<img src=x onerror="alert(1)">',
        category="<script>bad()</script>",
        difficulty="easy",
        model_id="model",
        scores=ScoreBreakdown(overall=1.0),
        passed=True,
    )
    report = format_report_html(
        {"count": 1, "passed": 1, "failed": 0, "overall": 1.0},
        [result],
        '<script>alert("x")</script>',
    )

    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in report
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in report
    assert "<script>bad()</script>" not in report
