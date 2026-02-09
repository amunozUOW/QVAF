#!/usr/bin/env python3
"""
Analysis Agent - Objective Dashboard Generator
===============================================
Generates HTML dashboard showing only objective metrics:
- AI correctness percentages
- RAG effect
- Question type distribution
- Confidence/consistency data

No judgments, risk levels, or recommendations.
"""

import json
import sys
import subprocess
import platform
from datetime import datetime
from pathlib import Path

# Import config for output paths
try:
    from config import DASHBOARDS_DIR, REPORTS_DIR, ensure_directories
    ensure_directories()
except ImportError:
    # Fallback for standalone usage - create proper directory structure
    DASHBOARDS_DIR = Path("./output/dashboards")
    REPORTS_DIR = Path("./output/reports")
    DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_report(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def safe_div(numerator, denominator, default=0.0):
    if denominator == 0:
        return default
    return numerator / denominator

def safe_pct(numerator, denominator, decimals=1):
    return round(safe_div(numerator, denominator, 0.0) * 100, decimals)

def calculate_grades(results, actual_scores=None):
    """Calculate AI correctness percentages from per-question data.

    Single data path: always computed from per-question results (authoritative
    since Option D uses Moodle's is_correct). actual_scores is kept in the
    report for reference but is no longer used for dashboard metrics.

    Returns dict with 'is_basic_mode' = True if only baseline scan was run.
    """
    correct_without = sum(1 for r in results if r.get('correct_without_rag'))
    total = len(results)

    # Detect basic mode from results (no RAG data available)
    has_rag_data = any(r.get('correct_with_rag') is not None for r in results)
    correct_with = sum(1 for r in results if r.get('correct_with_rag')) if has_rag_data else None

    grade_without = safe_pct(correct_without, total)
    grade_with = safe_pct(correct_with, total) if correct_with is not None else None

    return {
        'is_basic_mode': not has_rag_data,
        'total': total,
        'correct_without_rag': correct_without,
        'correct_with_rag': correct_with,
        'grade_without_rag': grade_without,
        'grade_with_rag': grade_with,
        'rag_effect': round(grade_with - grade_without, 1) if grade_with is not None else None
    }

def calculate_grades_by_type(results):
    """Calculate AI correctness by question type"""
    by_type = {}
    
    for r in results:
        qtype = r.get('question_type', 'UNKNOWN')
        if qtype not in by_type:
            by_type[qtype] = {'total': 0, 'correct_without': 0, 'correct_with': 0}
        
        by_type[qtype]['total'] += 1
        if r.get('correct_without_rag'):
            by_type[qtype]['correct_without'] += 1
        if r.get('correct_with_rag'):
            by_type[qtype]['correct_with'] += 1
    
    for qtype in by_type:
        t = by_type[qtype]['total']
        by_type[qtype]['grade_without'] = safe_pct(by_type[qtype]['correct_without'], t)
        by_type[qtype]['grade_with'] = safe_pct(by_type[qtype]['correct_with'], t)
        by_type[qtype]['rag_effect'] = round(by_type[qtype]['grade_with'] - by_type[qtype]['grade_without'], 1)
    
    return by_type

def count_by_correctness_pattern(results):
    """Count questions by AI correctness pattern.

    Handles basic mode (correct_with_rag is None) by counting questions
    as simply correct/incorrect using correct_both and incorrect_both buckets.
    """
    patterns = {
        'correct_both': 0,      # AI correct without and with RAG (or just correct in basic mode)
        'correct_rag_only': 0,  # AI correct only with RAG
        'correct_baseline_only': 0,  # AI correct only without RAG
        'incorrect_both': 0     # AI incorrect in both conditions (or just incorrect in basic mode)
    }

    for r in results:
        without = r.get('correct_without_rag')
        with_rag = r.get('correct_with_rag')

        if with_rag is None:
            # Basic mode — only baseline data, no pattern comparison possible
            if without:
                patterns['correct_both'] += 1
            else:
                patterns['incorrect_both'] += 1
        elif without and with_rag:
            patterns['correct_both'] += 1
        elif with_rag and not without:
            patterns['correct_rag_only'] += 1
        elif without and not with_rag:
            patterns['correct_baseline_only'] += 1
        else:
            patterns['incorrect_both'] += 1

    return patterns

def count_confidence_levels(results):
    """Count questions by AI confidence level"""
    levels = {'high': 0, 'medium': 0, 'low': 0}
    high_conf_incorrect = 0
    
    for r in results:
        # Check baseline confidence
        conf = r.get('confidence_without_rag', 0) or 0
        if conf >= 80:
            levels['high'] += 1
            if not r.get('correct_without_rag'):
                high_conf_incorrect += 1
        elif conf >= 50:
            levels['medium'] += 1
        else:
            levels['low'] += 1
    
    return levels, high_conf_incorrect

def generate_html_dashboard(report, source_file):
    """Generate objective HTML dashboard.

    Detects basic scan mode (no RAG) and generates appropriate layout.
    """

    summary = report['quantitative_summary']
    results = report['question_results']
    actual_scores = report.get('actual_scores')

    # Check scan mode from the data
    scan_mode = report.get('scan_mode', 'full')  # 'basic' or 'full'
    is_basic_mode = scan_mode == 'basic' or (actual_scores and actual_scores.get('with_rag_score') is None)

    total = summary['total_questions']
    grades = calculate_grades(results, actual_scores)
    grades_by_type = calculate_grades_by_type(results)
    patterns = count_by_correctness_pattern(results)
    confidence_levels, high_conf_incorrect = count_confidence_levels(results)

    # Override is_basic_mode from grades calculation
    if grades.get('is_basic_mode'):
        is_basic_mode = True
    
    # Check for consistency data (from multi-sample mode)
    has_consistency_data = any(r.get('consistency_without_rag') or r.get('consistency_with_rag') for r in results)
    
    # Calculate average consistency if available
    avg_consistency = None
    consistency_total = None
    if has_consistency_data:
        consistency_values = []
        for r in results:
            for key in ['consistency_without_rag', 'consistency_with_rag']:
                val = r.get(key)
                if val and '/' in str(val):
                    count, total_s = str(val).split('/')
                    consistency_values.append(int(count))
                    consistency_total = int(total_s)
        if consistency_values:
            avg_consistency = sum(consistency_values) / len(consistency_values)
    
    # Question type distribution
    type_dist = {}
    for r in results:
        qtype = r.get('question_type', 'UNKNOWN')
        type_dist[qtype] = type_dist.get(qtype, 0) + 1
    
    # Build grades by type table rows - different for basic vs full mode
    grades_rows = ""
    if is_basic_mode:
        for qtype, data in sorted(grades_by_type.items()):
            grades_rows += f"""
            <tr>
                <td>{qtype}</td>
                <td>{data['total']}</td>
                <td>{data['grade_without']}%</td>
            </tr>"""
    else:
        for qtype, data in sorted(grades_by_type.items()):
            grades_rows += f"""
            <tr>
                <td>{qtype}</td>
                <td>{data['total']}</td>
                <td>{data['grade_without']}%</td>
                <td>{data['grade_with']}%</td>
                <td>{data['rag_effect']:+.1f}%</td>
            </tr>"""

    # Build correctness pattern table rows - different for basic vs full mode
    if is_basic_mode:
        correct_count = patterns['correct_both'] + patterns['correct_baseline_only']
        incorrect_count = patterns['incorrect_both'] + patterns['correct_rag_only']
        pattern_rows = f"""
        <tr><td>AI Correct</td><td>{correct_count}</td><td>{safe_pct(correct_count, total)}%</td></tr>
        <tr><td>AI Incorrect</td><td>{incorrect_count}</td><td>{safe_pct(incorrect_count, total)}%</td></tr>
        """
    else:
        pattern_rows = f"""
        <tr><td>AI Correct (Both Conditions)</td><td>{patterns['correct_both']}</td><td>{safe_pct(patterns['correct_both'], total)}%</td></tr>
        <tr><td>AI Correct Only With Materials</td><td>{patterns['correct_rag_only']}</td><td>{safe_pct(patterns['correct_rag_only'], total)}%</td></tr>
        <tr><td>AI Correct Only Without Materials</td><td>{patterns['correct_baseline_only']}</td><td>{safe_pct(patterns['correct_baseline_only'], total)}%</td></tr>
        <tr><td>AI Incorrect (Both Conditions)</td><td>{patterns['incorrect_both']}</td><td>{safe_pct(patterns['incorrect_both'], total)}%</td></tr>
        """
    
    # Per-question details - different layout for basic vs full mode
    question_rows = ""
    for r in results:
        q_id = r.get('id', '?')
        q_type = r.get('question_type', 'UNKNOWN')
        correct_without = '✓' if r.get('correct_without_rag') else '✗'

        # Use consistency if available, otherwise confidence
        if r.get('consistency_without_rag'):
            metric_without = r.get('consistency_without_rag', '-')
        else:
            conf = r.get('confidence_without_rag', 0) or 0
            metric_without = f"{conf}%"

        if is_basic_mode:
            question_rows += f"""
            <tr>
                <td>{q_id}</td>
                <td>{q_type}</td>
                <td>{correct_without}</td>
                <td>{metric_without}</td>
            </tr>"""
        else:
            correct_with = '✓' if r.get('correct_with_rag') else '✗'
            if r.get('consistency_with_rag'):
                metric_with = r.get('consistency_with_rag', '-')
            else:
                conf = r.get('confidence_with_rag', 0) or 0
                metric_with = f"{conf}%"

            question_rows += f"""
            <tr>
                <td>{q_id}</td>
                <td>{q_type}</td>
                <td>{correct_without}</td>
                <td>{metric_without}</td>
                <td>{correct_with}</td>
                <td>{metric_with}</td>
            </tr>"""
    
    # Generate vulnerability analysis rows (objective, no traffic lights)
    qualitative = report.get('qualitative_analyses', [])
    vulnerability_analysis_rows = ""
    
    for qa in qualitative:
        q_id = qa.get('id', '?')
        vuln_cat = qa.get('vulnerability_category', 'UNKNOWN')
        q_type = qa.get('question_type', 'UNKNOWN')
        analysis = qa.get('analysis', 'No analysis available')
        
        # Clean up analysis text
        analysis_html = analysis.replace('\n', '<br>')
        
        vulnerability_analysis_rows += f"""
        <div style="border-left: 4px solid #ccc; padding: 15px; margin-bottom: 15px; background: #f8f9fa; border-radius: 4px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <strong>Question {q_id}</strong>
                <span style="color: #666;">{q_type}</span>
            </div>
            <div style="color: #555; font-size: 14px; line-height: 1.6;">
                {analysis_html}
            </div>
        </div>"""
    
    if not vulnerability_analysis_rows:
        vulnerability_analysis_rows = "<p style='color: #999;'>No detailed analysis available. Run reform_agent.py to generate.</p>"
    
    timestamp = datetime.now().strftime("%-d %B %Y")
    
    # Calculate average confidence across all answers
    all_confidences = []
    for r in results:
        conf_without = r.get('confidence_without_rag')
        conf_with = r.get('confidence_with_rag')
        if conf_without is not None and conf_without > 0:
            all_confidences.append(conf_without)
        if conf_with is not None and conf_with > 0:
            all_confidences.append(conf_with)
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    # Fifth metric box - show consistency if available, otherwise average confidence
    if avg_consistency is not None:
        fifth_metric_value = f"{avg_consistency:.1f}/{consistency_total}"
        fifth_metric_label = "Avg Consistency"
        fifth_metric_sublabel = "Multi-sample measurement"
        metric_header = "Consistency"
    else:
        fifth_metric_value = f"{avg_confidence:.0f}%"
        fifth_metric_label = "Avg Confidence"
        fifth_metric_sublabel = "How certain the AI was"
        metric_header = "Confidence"

    # Build metrics section based on mode
    if is_basic_mode:
        # Basic mode - fewer metrics, no course materials columns
        metrics_html = f"""
            <div class="metric-card">
                <div class="value">{grades['grade_without_rag']}%</div>
                <div class="label">AI Accuracy</div>
                <div class="sublabel">{grades['correct_without_rag']} of {grades['total']} correct</div>
            </div>
            <div class="metric-card">
                <div class="value">{total}</div>
                <div class="label">Questions Assessed</div>
                <div class="sublabel">&nbsp;</div>
            </div>
            <div class="metric-card">
                <div class="value">{fifth_metric_value}</div>
                <div class="label">{fifth_metric_label}</div>
                <div class="sublabel">{fifth_metric_sublabel}</div>
            </div>
        """
        # Table headers for basic mode
        grades_table_header = """
            <tr>
                <th>Question Type</th>
                <th>Count</th>
                <th>AI Accuracy</th>
            </tr>"""
        question_table_header = f"""
            <tr>
                <th>Q#</th>
                <th>Type</th>
                <th>Result</th>
                <th>{metric_header}</th>
            </tr>"""
        # Chart data - simpler for basic mode
        correct_count = patterns['correct_both'] + patterns['correct_baseline_only']
        incorrect_count = patterns['incorrect_both'] + patterns['correct_rag_only']
        chart_labels = "['AI Correct', 'AI Incorrect']"
        chart_data = f"[{correct_count}, {incorrect_count}]"
        chart_colors = "['#27ae60', '#e74c3c']"
    else:
        # Full mode - all metrics and course materials comparison
        metrics_html = f"""
            <div class="metric-card">
                <div class="value">{grades['grade_without_rag']}%</div>
                <div class="label">AI Accuracy (No Materials)</div>
                <div class="sublabel">{grades['correct_without_rag']} of {grades['total']} correct</div>
            </div>
            <div class="metric-card">
                <div class="value">{grades['grade_with_rag']}%</div>
                <div class="label">AI Accuracy (With Materials)</div>
                <div class="sublabel">{grades['correct_with_rag']} of {grades['total']} correct</div>
            </div>
            <div class="metric-card">
                <div class="value">{grades['rag_effect']:+.1f}%</div>
                <div class="label">Material Effect</div>
                <div class="sublabel">Change when given course materials</div>
            </div>
            <div class="metric-card">
                <div class="value">{total}</div>
                <div class="label">Questions Assessed</div>
                <div class="sublabel">&nbsp;</div>
            </div>
            <div class="metric-card">
                <div class="value">{fifth_metric_value}</div>
                <div class="label">{fifth_metric_label}</div>
                <div class="sublabel">{fifth_metric_sublabel}</div>
            </div>
        """
        # Table headers for full mode
        grades_table_header = """
            <tr>
                <th>Question Type</th>
                <th>Count</th>
                <th>No Materials</th>
                <th>With Materials</th>
                <th>Material Effect</th>
            </tr>"""
        question_table_header = f"""
            <tr>
                <th>Q#</th>
                <th>Type</th>
                <th>No Materials</th>
                <th>{metric_header}</th>
                <th>With Materials</th>
                <th>{metric_header}</th>
            </tr>"""
        # Chart data for full mode
        chart_labels = "['AI Correct (Both Conditions)', 'AI Correct Only With Materials', 'AI Correct Only Without Materials', 'AI Incorrect (Both Conditions)']"
        chart_data = f"[{patterns['correct_both']}, {patterns['correct_rag_only']}, {patterns['correct_baseline_only']}, {patterns['incorrect_both']}]"
        chart_colors = "['#3498db', '#f39c12', '#9b59b6', '#27ae60']"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quiz Vulnerability Assessment Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        
        header {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}
        header h1 {{ font-size: 24px; margin-bottom: 10px; }}
        header .purpose {{
            opacity: 0.9;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        header .date {{ opacity: 0.6; font-size: 12px; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-card .label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
            margin-top: 5px;
        }}
        .metric-card .sublabel {{
            font-size: 11px;
            color: #999;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-card h2 {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        .chart-container {{ height: 250px; }}
        
        .table-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .table-section h2 {{
            font-size: 16px;
            margin-bottom: 15px;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            color: #666;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Quiz Vulnerability Assessment Report</h1>
            <p class="purpose">
                This report measures how effectively an AI was able to answer your quiz questions.
                It identifies which questions the AI found easy or difficult, and provides analysis
                to help you understand why. Use these findings to inform your assessment design.
                This report does not AI-proof your assessment, and the analysis is intended to
                support your professional judgment, not replace it.
            </p>
            <p class="date">Generated: {timestamp}</p>
        </header>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            {metrics_html}
        </div>
        
        <!-- Charts -->
        <div class="charts-row">
            <div class="chart-card">
                <h2>How the AI Performed</h2>
                <div class="chart-container">
                    <canvas id="patternChart"></canvas>
                </div>
            </div>
            <div class="chart-card">
                <h2>Question Types in Your Quiz</h2>
                <div class="chart-container">
                    <canvas id="typeChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Correctness Pattern Table -->
        <div class="table-section">
            <h2>AI Results Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Result</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    {pattern_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Performance by Question Type -->
        <div class="table-section">
            <h2>AI Accuracy by Question Type</h2>
            <table>
                <thead>
                    {grades_table_header}
                </thead>
                <tbody>
                    {grades_rows}
                </tbody>
            </table>
        </div>

        <!-- Per-Question Details -->
        <div class="table-section">
            <h2>Question-by-Question Breakdown</h2>
            <table>
                <thead>
                    {question_table_header}
                </thead>
                <tbody>
                    {question_rows}
                </tbody>
            </table>
        </div>
        
        <!-- Vulnerability Analysis Section -->
        <div class="table-section">
            <h2>Per-Question Vulnerability Analysis</h2>
            <p style="color: #666; margin-bottom: 15px;">What made each question easy or difficult for the AI, and what you could change</p>
            {vulnerability_analysis_rows}
        </div>
        
        <footer>
            Generated by the Quiz Vulnerability Assessment Framework (QVAF)
        </footer>
    </div>
    
    <script>
        // Correctness Pattern Chart
        const patternCtx = document.getElementById('patternChart').getContext('2d');
        new Chart(patternCtx, {{
            type: 'doughnut',
            data: {{
                labels: {chart_labels},
                datasets: [{{
                    data: {chart_data},
                    backgroundColor: {chart_colors},
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
        
        // Question Type Chart
        const typeCtx = document.getElementById('typeChart').getContext('2d');
        new Chart(typeCtx, {{
            type: 'doughnut',
            data: {{
                labels: {list(type_dist.keys())},
                datasets: [{{
                    data: {list(type_dist.values())},
                    backgroundColor: ['#3498db', '#1abc9c', '#e67e22', '#9b59b6', '#34495e'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def generate_markdown_report(report, source_file):
    """Generate simple markdown report with objective data"""

    summary = report['quantitative_summary']
    results = report['question_results']
    actual_scores = report.get('actual_scores')

    grades = calculate_grades(results, actual_scores)
    patterns = count_by_correctness_pattern(results)

    # Check if basic mode (no RAG data)
    is_basic_mode = grades.get('is_basic_mode', False)

    md_timestamp = datetime.now().strftime("%-d %B %Y")

    purpose_text = """This report measures how effectively an AI was able to answer your quiz questions.
It identifies which questions the AI found easy or difficult, and provides analysis
to help you understand why. Use these findings to inform your assessment design.
This report does not AI-proof your assessment, and the analysis is intended to
support your professional judgment, not replace it."""

    if is_basic_mode:
        # Basic mode markdown - no course materials columns
        md = f"""# Quiz Vulnerability Assessment Report

{purpose_text}

Generated: {md_timestamp}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Questions Assessed | {grades['total']} |
| AI Accuracy | {grades['grade_without_rag']}% ({grades['correct_without_rag']} of {grades['total']}) |

## AI Results Summary

| Result | Count | Percentage |
|--------|-------|------------|
| AI Correct | {patterns['correct_both'] + patterns['correct_baseline_only']} | {safe_pct(patterns['correct_both'] + patterns['correct_baseline_only'], grades['total'])}% |
| AI Incorrect | {patterns['incorrect_both'] + patterns['correct_rag_only']} | {safe_pct(patterns['incorrect_both'] + patterns['correct_rag_only'], grades['total'])}% |

## Question-by-Question Breakdown

| Q# | Type | Result |
|----|------|--------|
"""
        for r in results:
            q_id = r.get('id', '?')
            q_type = r.get('question_type', 'UNKNOWN')
            correct_without = '✓' if r.get('correct_without_rag') else '✗'
            md += f"| {q_id} | {q_type} | {correct_without} |\n"
    else:
        # Full mode markdown - includes course materials comparison
        md = f"""# Quiz Vulnerability Assessment Report

{purpose_text}

Generated: {md_timestamp}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Questions Assessed | {grades['total']} |
| AI Accuracy (No Materials) | {grades['grade_without_rag']}% ({grades['correct_without_rag']} of {grades['total']}) |
| AI Accuracy (With Materials) | {grades['grade_with_rag']}% ({grades['correct_with_rag']} of {grades['total']}) |
| Material Effect | {grades['rag_effect']:+.1f}% |

## AI Results Summary

| Result | Count | Percentage |
|--------|-------|------------|
| AI Correct (Both Conditions) | {patterns['correct_both']} | {safe_pct(patterns['correct_both'], grades['total'])}% |
| AI Correct Only With Materials | {patterns['correct_rag_only']} | {safe_pct(patterns['correct_rag_only'], grades['total'])}% |
| AI Correct Only Without Materials | {patterns['correct_baseline_only']} | {safe_pct(patterns['correct_baseline_only'], grades['total'])}% |
| AI Incorrect (Both Conditions) | {patterns['incorrect_both']} | {safe_pct(patterns['incorrect_both'], grades['total'])}% |

## Question-by-Question Breakdown

| Q# | Type | No Materials | With Materials |
|----|------|--------------|----------------|
"""
        for r in results:
            q_id = r.get('id', '?')
            q_type = r.get('question_type', 'UNKNOWN')
            correct_without = '✓' if r.get('correct_without_rag') else '✗'
            correct_with = '✓' if r.get('correct_with_rag') else '✗'
            md += f"| {q_id} | {q_type} | {correct_without} | {correct_with} |\n"

    return md


def open_in_browser(filepath):
    """Open the dashboard in the default browser"""
    system = platform.system()
    try:
        if system == 'Darwin':
            subprocess.run(['open', filepath])
        elif system == 'Windows':
            subprocess.run(['start', filepath], shell=True)
        else:
            subprocess.run(['xdg-open', filepath])
        print(f"Dashboard opened in browser")
    except Exception as e:
        print(f"Could not open browser: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analysis_agent.py <report.json>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Loading {filepath}...")
    report = load_report(filepath)

    # Generate markdown report
    print("Generating markdown report...")
    md_report = generate_markdown_report(report, filepath)

    # Determine output paths - keep reports in same dir as input, dashboards in DASHBOARDS_DIR
    input_path = Path(filepath)
    base_name = input_path.stem.replace('_analysis_report', '').replace('_vulnerability_report', '')

    # Markdown goes with the report
    md_output = input_path.parent / f"{base_name}_analysis_summary.md"

    # Dashboard goes to dashboards directory - ensure it exists
    DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)
    html_output = DASHBOARDS_DIR / f"{base_name}_dashboard.html"

    print(f"DEBUG: DASHBOARDS_DIR = {DASHBOARDS_DIR.absolute()}")
    print(f"DEBUG: html_output = {html_output.absolute()}")

    with open(md_output, 'w') as f:
        f.write(md_report)
    print(f"Markdown report saved to {md_output}")

    # Generate HTML dashboard
    print("Generating HTML dashboard...")
    html = generate_html_dashboard(report, filepath)
    with open(html_output, 'w') as f:
        f.write(html)
    print(f"Dashboard saved to {html_output}")

    # Open in browser
    open_in_browser(html_output)

    print("\nDone.")