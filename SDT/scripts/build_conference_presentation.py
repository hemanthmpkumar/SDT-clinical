#!/usr/bin/env python3
"""
scripts/build_conference_presentation.py

Populate ICI3T_2026_PPT_Template_Adroid.pptx with the complete, publication-grade
presentation contents derived from main_ijds_blind.pdf / main_ijds_blind.tex
for presentation at the 2026 International Conference on Intelligent Computing,
Information & Innovative Data Technologies (ICI3T 2026).
"""

import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# Color Palette Definitions
NAVY_PRIMARY = RGBColor(16, 44, 87)       # Deep Navy Blue for Titles & Headers
NAVY_ACCENT = RGBColor(24, 76, 120)       # Accent Blue for Subtitles & Badges
SLATE_DARK = RGBColor(33, 37, 41)         # Body text dark
SLATE_MUTED = RGBColor(108, 117, 125)     # Secondary text / metadata
CARD_BG = RGBColor(248, 250, 253)         # Very light cool blue/gray card fill
CARD_BORDER = RGBColor(215, 226, 238)     # Soft border
HIGHLIGHT_BG = RGBColor(235, 243, 250)    # Active method / highlight fill
HIGHLIGHT_TEXT = RGBColor(12, 60, 105)    # Highlight text
WHITE = RGBColor(255, 255, 255)
SUCCESS_DARK = RGBColor(20, 108, 67)
CRIMSON_DARK = RGBColor(140, 29, 24)

FONT_PRIMARY = "Segoe UI"

def clear_slide_content(slide):
    """Remove Subtitle 2 placeholder or any non-banner shape from slide."""
    for s in list(slide.shapes):
        if s.name != "Picture 1" and s.name != "Picture 3":
            sp = s._element
            sp.getparent().remove(sp)

def add_header(slide, title_text, category_tag="ICI3T 2026 | TECHNICAL PRESENTATION"):
    """Add standardized top header with category tag and accent underline."""
    tb = slide.shapes.add_textbox(Inches(0.60), Inches(1.68), Inches(12.13), Inches(0.60))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_tag = tf.paragraphs[0]
    p_tag.text = category_tag.upper()
    p_tag.font.name = FONT_PRIMARY
    p_tag.font.size = Pt(9.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = NAVY_ACCENT
    p_tag.space_after = Pt(2)

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = FONT_PRIMARY
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_PRIMARY

    # Accent underline
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(2.32), Inches(12.13), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = CARD_BORDER
    line.line.color.rgb = CARD_BORDER

def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER, rounded=True):
    """Add a styled background container card."""
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    card = slide.shapes.add_shape(shape_type, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    card.line.color.rgb = border_color
    card.line.width = Pt(1)
    return card

def add_para(tf, text, font_size=12, bold=False, color=SLATE_DARK, space_after=Pt(3), space_before=Pt(0), bullet=False):
    """Helper to add formatted paragraph."""
    p = tf.add_paragraph() if len(tf.paragraphs) > 0 and tf.paragraphs[0].text else tf.paragraphs[0]
    p.text = ("• " if bullet else "") + text
    p.font.name = FONT_PRIMARY
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.space_after = space_after
    p.space_before = space_before
    return p

# -----------------------------------------------------------------------------
# SLIDE BUILDERS
# -----------------------------------------------------------------------------

def build_slide_1_title(prs):
    slide = prs.slides[0]
    clear_slide_content(slide)

    # Main Card
    add_card(slide, Inches(0.60), Inches(1.80), Inches(12.13), Inches(5.35), bg_color=CARD_BG, border_color=CARD_BORDER)

    tb = slide.shapes.add_textbox(Inches(0.90), Inches(2.00), Inches(11.53), Inches(4.95))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    # Conference Sub-Tag
    p0 = add_para(tf, "INTERNATIONAL CONFERENCE ON INTELLIGENT COMPUTING, INFORMATION & INNOVATIVE DATA TECHNOLOGIES (ICI3T 2026)",
                  font_size=11, bold=True, color=NAVY_ACCENT, space_after=Pt(8))

    # Paper Title
    p1 = add_para(tf, "Spectral Diagnostic Trajectories: Multi-Modal Hypergraph State-Space Tracking, Context Decoupling, and Congestion Mitigation in Stochastic Clinical Service Systems",
                  font_size=23, bold=True, color=NAVY_PRIMARY, space_after=Pt(14))

    # Accent divider
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.90), Inches(3.68), Inches(2.5), Inches(0.04))
    div.fill.solid()
    div.fill.fore_color.rgb = NAVY_ACCENT
    div.line.color.rgb = NAVY_ACCENT

    # Authors Box
    tb_auth = slide.shapes.add_textbox(Inches(0.90), Inches(3.85), Inches(11.53), Inches(2.0))
    tf_auth = tb_auth.text_frame
    tf_auth.word_wrap = True
    tf_auth.margin_left = tf_auth.margin_top = tf_auth.margin_right = tf_auth.margin_bottom = 0

    p_a1 = add_para(tf_auth, "Hemanth Manchabale Papachappa", font_size=14, bold=True, color=NAVY_PRIMARY, space_after=Pt(1))
    add_para(tf_auth, "Graduate School of Engineering / Biomedical Informatics Program, Stanford University, Stanford, CA 94305, USA",
             font_size=11, bold=False, color=SLATE_MUTED, space_after=Pt(6))

    p_a2 = add_para(tf_auth, "Aniruddha Ganguly", font_size=14, bold=True, color=NAVY_PRIMARY, space_after=Pt(1))
    add_para(tf_auth, "Department of Computer Science and Engineering, Jadavpur University, Kolkata, West Bengal 700032, India",
             font_size=11, bold=False, color=SLATE_MUTED, space_after=Pt(12))

    # Track / Venue Footer inside card
    p_foot = add_para(tf_auth, "Track: Healthcare AI, Spectral Graph Theory & Stochastic Operations  |  Organiser: \"Gheorghe Asachi\" Technical University of Iasi, Romania",
                      font_size=10.5, bold=True, color=NAVY_ACCENT)


def build_slide_2_index(prs):
    slide = prs.slides[1]
    clear_slide_content(slide)
    add_header(slide, "1. INDEX: Presentation Roadmap", "AGENDA & STRUCTURE")

    # 3 Columns of 3 Cards = 9 structured presentation cards
    cards_data = [
        # Col 1
        [
            ("01. Operational Motivation", "High-acuity emergency care, multimodal telemetry fragmentation, and search latency bottlenecks.", NAVY_PRIMARY),
            ("02. Literature Review", "Systematic synthesis of lexical, dense RAG, knowledge graphs, and Riemannian manifold approaches.", NAVY_PRIMARY),
            ("03. Critical Research Gaps", "The memoryless vs. over-smoothed dilemma, waveform neglect, and context contamination.", CRIMSON_DARK)
        ],
        # Col 2
        [
            ("04. The SDT Framework", "Dynamic hypergraph Laplacians, continuous state-space Mamba-2 waveforms (125 Hz).", NAVY_ACCENT),
            ("05. Theoretical Decoupling", "Theorem 1: Linear-time O(|Et|) pivot detection; Theorem 2: Formal Cheeger conductance bounds.", NAVY_ACCENT),
            ("06. Cognitive Prefetching", "Spectrally biased random walks under Sweller working-memory surprisal limits (Cmax = 7.0 bits).", NAVY_ACCENT)
        ],
        # Col 3
        [
            ("07. Erlang C Queueing Model", "Theorem 3: Super-linear waiting time sensitivity O((1-ρ)^-2) in congested hospital servers.", SUCCESS_DARK),
            ("08. Benchmark Results", "9-arm Latin-square trial on 1,000 vignettes (9,000 runs): TTCI (-28.9%), Accuracy (93.9%), Recall@5.", SUCCESS_DARK),
            ("09. Operational Translation", "NASA-TLX workload reductions, >10,200 annual ER boarding hours saved, and deployment roadmap.", SUCCESS_DARK)
        ]
    ]

    col_width = Inches(3.85)
    col_gap = Inches(0.29)
    card_h = Inches(1.38)
    card_gap = Inches(0.18)

    for c_idx, col in enumerate(cards_data):
        c_left = Inches(0.60) + c_idx * (col_width + col_gap)
        for r_idx, (num_title, desc, tag_color) in enumerate(col):
            c_top = Inches(2.48) + r_idx * (card_h + card_gap)
            add_card(slide, c_left, c_top, col_width, card_h)

            tb = slide.shapes.add_textbox(c_left + Inches(0.18), c_top + Inches(0.14), col_width - Inches(0.36), card_h - Inches(0.28))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

            add_para(tf, num_title, font_size=12.5, bold=True, color=tag_color, space_after=Pt(3))
            add_para(tf, desc, font_size=10.5, bold=False, color=SLATE_DARK, space_after=Pt(0))


def build_slide_3_abstract(prs):
    slide = prs.slides[2]
    clear_slide_content(slide)
    add_header(slide, "2. ABSTRACT: Executive Summary & Theoretical Contributions", "EXECUTIVE SUMMARY")

    # 3 Horizontal thematic cards
    sections = [
        ("The Clinical Dilemma",
         "• Severe Time Sink: Clinicians spend >33% of active shifts querying fragmented EHR charts.\n"
         "• Trajectory Failure: Systems either forget history (stateless) or lag behind shifts (Riemannian smoothing).\n"
         "• Context Contamination: Obsolete hypotheses pollute prompts during acute pivots, causing cognitive fatigue and boarding delays.",
         CRIMSON_DARK),
        
        ("The SDT Framework",
         "• Multi-Modal Tracking: Unifies 125 Hz telemetry (Mamba-2) and clinical notes into dynamic hypergraphs.\n"
         "• Cheeger Decoupling Cut: Monitors Laplacian spectral gap in linear time O(|Et|); isolates obsolete context (P_leak ≤ √(2τ_pivot)).\n"
         "• Cognitive Prefetch: Stages prospective records via 0/1 knapsack bounded by Sweller limits (Cmax = 7.0 bits).",
         NAVY_PRIMARY),

        ("Empirical & Queueing Impact",
         "• 28.9% Faster Insight: TTCI drops to 62.97s (vs. 88.52s control, p < 10^-12) with 93.9% resolution accuracy.\n"
         "• 32.2% Cognitive Relief: Modeled workload drops; mental demand -42.5%, frustration -56.2%.\n"
         "• Macro ED Impact: Super-linear queue leverage saves 12.31 min wait time/patient, eliminating >10,200 annual boarding hours.",
         SUCCESS_DARK)
    ]

    card_w = Inches(3.85)
    card_h = Inches(4.55)
    gap = Inches(0.29)

    for i, (title, body, header_color) in enumerate(sections):
        left = Inches(0.60) + i * (card_w + gap)
        top = Inches(2.48)
        add_card(slide, left, top, card_w, card_h)

        tb = slide.shapes.add_textbox(left + Inches(0.20), top + Inches(0.24), card_w - Inches(0.40), card_h - Inches(0.48))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        add_para(tf, title, font_size=13, bold=True, color=header_color, space_after=Pt(10))
        for line in body.split("\n"):
            line = line.strip()
            if line:
                is_bullet = line.startswith("•")
                text = line[1:].strip() if is_bullet else line
                add_para(tf, text, font_size=10.5, bold=False, color=SLATE_DARK, space_after=Pt(8), bullet=is_bullet)


def build_slide_4_introduction(prs):
    slide = prs.slides[3]
    clear_slide_content(slide)
    add_header(slide, "3. INTRODUCTION: High-Acuity Healthcare Operations & Context Drift", "OPERATIONAL MOTIVATION")

    pillars = [
        ("Information Overload & Latency",
         "• High-Acuity Reality: 20–30 concurrent, undifferentiated patients per emergency physician.\n"
         "• Telemetry Fragmentation: Continuous 125–250 Hz waveforms, discrete labs, imaging, and notes.\n"
         "• Search Bottleneck: >33% shift time spent in chart navigation; search latency directly delays critical care.",
         NAVY_PRIMARY),

        ("Diagnostic Regime Shifts",
         "• Non-Stationary Workflows: 53.1% of complex emergency cases exhibit acute diagnostic bifurcations.\n"
         "• Acute Pivot Example: Pneumonia working impression abruptly refuted by emergent telemetry indicating pulmonary embolism.\n"
         "• Context Poisoning: Legacy search retains stale evidence in context, misguiding retrieval and inducing cognitive closure.",
         CRIMSON_DARK),

        ("Congested Queueing Dynamics",
         "• Near-Saturation Operations: Emergency departments operate at heavy utilization (ρ = λ/(cμ) > 0.90).\n"
         "• Super-Linear Delays: Near saturation, queue delay scales quadratically: ∂Wq/∂(Δt) ≈ O((1-ρ)^-2).\n"
         "• Compounding Leverage: Saving 25.6s of search time pulls the hospital away from catastrophic waiting asymptotes.",
         SUCCESS_DARK)
    ]

    card_w = Inches(3.85)
    card_h = Inches(4.55)
    gap = Inches(0.29)

    for i, (title, body, header_color) in enumerate(pillars):
        left = Inches(0.60) + i * (card_w + gap)
        top = Inches(2.48)
        add_card(slide, left, top, card_w, card_h)

        tb = slide.shapes.add_textbox(left + Inches(0.20), top + Inches(0.24), card_w - Inches(0.40), card_h - Inches(0.48))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        add_para(tf, title, font_size=13, bold=True, color=header_color, space_after=Pt(10))
        for line in body.split("\n"):
            line = line.strip()
            if line:
                is_bullet = line.startswith("•")
                text = line[1:].strip() if is_bullet else line
                add_para(tf, text, font_size=10.5, bold=False, color=SLATE_DARK, space_after=Pt(8), bullet=is_bullet)



def build_slide_5_literature(prs):
    slide = prs.slides[4]
    clear_slide_content(slide)
    add_header(slide, "4. LITERATURE REVIEW: Systematic Synthesis of Clinical Retrieval Paradigms", "LITERATURE COMPARISON")

    # Note banner
    note_tb = slide.shapes.add_textbox(Inches(0.60), Inches(2.35), Inches(12.13), Inches(0.30))
    p_note = note_tb.text_frame.paragraphs[0]
    p_note.text = "Comprehensive evaluation of existing paradigms across mathematical representation, pivot adaptation, waveform support, and operational guarantees:"
    p_note.font.name = FONT_PRIMARY
    p_note.font.size = Pt(10)
    p_note.font.color.rgb = SLATE_MUTED

    # Table: 6 rows (header + 5 rows), 5 columns
    rows, cols = 6, 5
    top, left = Inches(2.70), Inches(0.60)
    width, height = Inches(12.13), Inches(4.30)

    t_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    t = t_shape.table

    # Column widths
    t.columns[0].width = Inches(2.10)
    t.columns[1].width = Inches(2.00)
    t.columns[2].width = Inches(2.40)
    t.columns[3].width = Inches(2.30)
    t.columns[4].width = Inches(3.33)

    headers = ["Retrieval Paradigm", "Benchmark Systems", "Mathematical Representation", "Diagnostic Pivot Adaptation", "Core Technical Limitations"]
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_PRIMARY
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    data = [
        ("Lexical Matching", "BM25, TF-IDF\n(Baeza-Yates '99)", "Inverted index, term frequency vectors", "None (Stateless query-by-query)", "High IDF over-weights rare historical terms; zero session continuity; high cognitive burden."),
        ("Dense Clinical RAG", "MedCPT (Jin '23),\nColBERTv2 ('22)", "Dual-encoder embeddings, MaxSim late interaction", "None (Memoryless dense lookup)", "Discards longitudinal interaction trajectory; forces repetitive manual query reformulations."),
        ("Static Knowledge Graphs", "GraphCare\n(Jiang et al. '24)", "Predefined medical ontology, 2-hop GNN message passing", "Static graph traversal", "Static relational graph cannot ingest continuous 125 Hz telemetry; inflexible to sudden bifurcations."),
        ("Continuous Manifolds", "GDT Framework\n(Riemannian SPD)", "Covariance on SPD manifold S_d^+ (Pennec '06)", "Continuous exponential smoothing (e^-γt)", "Riemannian geodesic smoothing lags behind abrupt discrete bifurcations; context contamination persists."),
        ("SDT (Proposed)", "Spectral Diagnostic Trajectories (Ours)", "Dynamic Hypergraph Laplacian + Mamba-2 State Space", "Exact Cheeger Cut on Fiedler Vector (v2)", "Resolves all prior limitations: O(|Et|) pivot detection, provable Cheeger bounds, super-linear queue relief.")
    ]

    for i, row in enumerate(data):
        is_ours = (i == 4)
        bg = HIGHLIGHT_BG if is_ours else (WHITE if i % 2 == 0 else CARD_BG)
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(9.5)
            p.font.bold = is_ours or (j == 0)
            p.font.color.rgb = HIGHLIGHT_TEXT if is_ours else SLATE_DARK
            p.alignment = PP_ALIGN.LEFT


def build_slide_6_gaps(prs):
    slide = prs.slides[5]
    clear_slide_content(slide)
    add_header(slide, "5. RESEARCH GAPS: Fundamental Deficiencies in Existing Paradigms", "CRITICAL RESEARCH GAPS")

    gaps = [
        ("GAP 01: Memoryless vs. Over-Smoothed Trajectory Dilemma",
         "• Stateless RAG: Discards longitudinal history entirely, forcing repetitive query reformulations.\n"
         "• Riemannian Manifolds (GDT): Enforces continuous exponential smoothing, lagging behind acute discrete bifurcations.",
         NAVY_PRIMARY),

        ("GAP 02: Neglect of High-Frequency Telemetry Waveforms",
         "• Text/Code Bias: Clinical RAG indexes static narratives, completely ignoring 125–250 Hz physiological telemetry.\n"
         "• Missed Early Signals: Waveforms carry the earliest markers of instability (e.g., S1Q3T3 RV strain in pulmonary embolism).",
         CRIMSON_DARK),

        ("GAP 03: Unbounded Context Contamination & Mental Fatigue",
         "• Context Poisoning: Obsolete hypotheses pollute prompts during pivots, inducing confirmation bias.\n"
         "• Unbounded Speculation: Prefetchers push uncurated records, breaching human working memory limits (Miller's 7±2).",
         NAVY_PRIMARY),

        ("GAP 04: Micro-Latency to Macro-Congestion Disconnect",
         "• Isolated Benchmarks: Models are evaluated purely on algorithm latency (ms) without clinical queueing context.\n"
         "• Super-Linear Queue Physics: Near saturation (ρ > 0.90), 25s task savings prevents hours of department-wide boarding.",
         SUCCESS_DARK)
    ]

    card_w = Inches(5.90)
    card_h = Inches(2.15)
    gap_x = Inches(0.33)
    gap_y = Inches(0.20)

    for idx, (title, body, tag_color) in enumerate(gaps):
        col = idx % 2
        row = idx // 2
        left = Inches(0.60) + col * (card_w + gap_x)
        top = Inches(2.48) + row * (card_h + gap_y)

        add_card(slide, left, top, card_w, card_h)

        tb = slide.shapes.add_textbox(left + Inches(0.20), top + Inches(0.18), card_w - Inches(0.40), card_h - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        add_para(tf, title, font_size=12, bold=True, color=tag_color, space_after=Pt(6))
        for line in body.split("\n"):
            line = line.strip()
            if line:
                is_bullet = line.startswith("•")
                text = line[1:].strip() if is_bullet else line
                add_para(tf, text, font_size=10.2, bold=False, color=SLATE_DARK, space_after=Pt(4), bullet=is_bullet)


def build_slide_7_methodology(prs):
    slide = prs.slides[6]
    clear_slide_content(slide)
    add_header(slide, "6. PROPOSED METHODOLOGY: The Spectral Diagnostic Trajectories (SDT) Framework", "SYSTEM ARCHITECTURE")

    # 5 Process Blocks (Flowchart layout across width)
    stages = [
        ("1. Multi-Modal Ingestion",
         "• Clinical Notes: LayerNorm ClinicalBERT (d=768→128)\n"
         "• Waveforms: 125 Hz continuous streams via linear-time Mamba-2 (d=128, N=64)\n"
         "• Discrete Events: Lab flowsheets & diagnostic codes",
         NAVY_PRIMARY),

        ("2. Dynamic Hypergraph",
         "• State Representation: Polyadic hyperedges e = (v_diag, v_wave, v_lab)\n"
         "• Composite Weight: Semantic + Temporal (e^-λΔt) + Severity\n"
         "• Laplacian: Lt = I - Dv^-1/2 At Dv^-1/2",
         NAVY_PRIMARY),

        ("3. Pivot Detection",
         "• Linear-Time O(|Et|): Track Fiedler value λ2(Lt) via warm-started Lanczos\n"
         "• Pivot Trigger: Spectral gap drop Δλ2(t) > τ_pivot = 0.15\n"
         "• Real-Time: Sub-15ms detection of acute diagnostic shifts",
         CRIMSON_DARK),

        ("4. Cheeger Decoupling",
         "• Chebyshev Filter: Degree K=3 polynomial smoothing\n"
         "• Cheeger Bisection: Partition into V_stale vs. V_active\n"
         "• Theorem 1 Bound: Provable leak bound P_leak ≤ √(2τ_pivot) = 0.548",
         CRIMSON_DARK),

        ("5. Cognitive Prefetch",
         "• Biased Random Walk: Prospective traversal on global graph (N=546k)\n"
         "• 0/1 Knapsack: Bounded by Sweller capacity (Cmax ≤ 7.0 bits)\n"
         "• M/M/c Queue Coupling: Compounds into macro wait reduction",
         SUCCESS_DARK)
    ]

    card_w = Inches(2.28)
    card_h = Inches(4.55)
    gap = Inches(0.18)

    for i, (title, body, color) in enumerate(stages):
        left = Inches(0.60) + i * (card_w + gap)
        top = Inches(2.48)

        # Highlight stage 3 & 4 (core innovations)
        is_highlight = (i in [2, 3])
        bg = HIGHLIGHT_BG if is_highlight else CARD_BG
        border = NAVY_ACCENT if is_highlight else CARD_BORDER

        add_card(slide, left, top, card_w, card_h, bg_color=bg, border_color=border)

        tb = slide.shapes.add_textbox(left + Inches(0.12), top + Inches(0.18), card_w - Inches(0.24), card_h - Inches(0.36))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        add_para(tf, title, font_size=11.5, bold=True, color=color, space_after=Pt(8))
        for line in body.split("\n"):
            line = line.strip()
            if line:
                is_bullet = line.startswith("•")
                text = line[1:].strip() if is_bullet else line
                add_para(tf, text, font_size=9.5, bold=False, color=SLATE_DARK, space_after=Pt(5), bullet=is_bullet)


def build_slide_8_results(prs):
    slide = prs.slides[7]
    clear_slide_content(slide)
    add_header(slide, "7. RESULTS & DISCUSSION: Primary Retrieval Outcomes on 1,000 Vignettes", "EXPERIMENTAL EVALUATION")

    # Left: Table across 9 experimental arms
    t_left, t_top, t_w, t_h = Inches(0.60), Inches(2.48), Inches(6.80), Inches(4.55)
    rows, cols = 10, 7
    t_shape = slide.shapes.add_table(rows, cols, t_left, t_top, t_w, t_h)
    t = t_shape.table

    # Widths
    t.columns[0].width = Inches(1.50)
    t.columns[1].width = Inches(0.95)
    t.columns[2].width = Inches(0.85)
    t.columns[3].width = Inches(0.85)
    t.columns[4].width = Inches(0.80)
    t.columns[5].width = Inches(0.95)
    t.columns[6].width = Inches(0.90)

    col_names = ["Condition", "TTCI (s) ↓", "Accuracy ↑", "Recall@5 ↑", "MRR ↑", "Cog Load ↓", "Latency (ms) ↓"]
    for j, name in enumerate(col_names):
        cell = t.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.text = name
        p.font.name = FONT_PRIMARY
        p.font.size = Pt(8.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

    results_data = [
        ("Control (TF-IDF)", "88.52 ± 2.01", "86.8%", "0.682", "0.541", "51.60 ± 0.42", "124.52 ± 0.28"),
        ("BM25", "92.23 ± 2.14", "84.6%", "0.664", "0.522", "53.07 ± 0.43", "125.10 ± 0.28"),
        ("LLM-RAG", "89.18 ± 2.36", "78.6%", "0.712", "0.589", "43.05 ± 0.38", "164.96 ± 0.34"),
        ("ColBERT-Clinical", "79.95 ± 2.06", "85.5%", "0.784", "0.672", "42.29 ± 0.35", "157.26 ± 0.33"),
        ("CMA (Dense RAG)", "78.71 ± 1.90", "86.8%", "0.801", "0.698", "43.43 ± 0.33", "146.46 ± 0.32"),
        ("MedCPT", "76.36 ± 1.86", "87.5%", "0.819", "0.716", "42.36 ± 0.32", "142.21 ± 0.31"),
        ("GraphCare", "75.62 ± 1.84", "87.1%", "0.825", "0.724", "41.44 ± 0.33", "149.46 ± 0.32"),
        ("GDT (Manifold)", "68.96 ± 1.57", "91.5%", "0.851", "0.774", "39.47 ± 0.30", "128.24 ± 0.28"),
        ("SDT (Ours)", "62.97 ± 1.40", "93.9%", "0.914", "0.862", "34.98 ± 0.28", "112.87 ± 0.25")
    ]

    for i, row in enumerate(results_data):
        is_ours = (i == 8)
        bg = HIGHLIGHT_BG if is_ours else (WHITE if i % 2 == 0 else CARD_BG)
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(8.5)
            p.font.bold = is_ours
            p.font.color.rgb = HIGHLIGHT_TEXT if is_ours else SLATE_DARK
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    # Right: Insert Figure (fig_ttci_bar.png) + Callout Cards
    fig_path = Path("latex/IJDS/figures/fig_ttci_bar.png")
    if fig_path.exists():
        slide.shapes.add_picture(str(fig_path), Inches(7.60), Inches(2.48), Inches(5.13), Inches(2.80))

    # Callout Card below figure
    add_card(slide, Inches(7.60), Inches(5.38), Inches(5.13), Inches(1.65))
    tb_c = slide.shapes.add_textbox(Inches(7.75), Inches(5.48), Inches(4.83), Inches(1.45))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0

    add_para(tf_c, "Key Empirical Findings (9,000 Evaluated Sessions):", font_size=11, bold=True, color=NAVY_PRIMARY, space_after=Pt(3))
    add_para(tf_c, "• 28.9% TTCI Speedup: 25.55s reduction over Control; 8.7% over GDT (p < 10^-12, d = 1.38).", font_size=9.5, space_after=Pt(2))
    add_para(tf_c, "• State-of-the-Art Accuracy: 93.9% resolution rate (+7.1% over Control, +2.4% over GDT).", font_size=9.5, space_after=Pt(2))
    add_para(tf_c, "• Recall@5 (91.4%) & MRR (0.862): Surpasses all dense and manifold baselines.", font_size=9.5, space_after=Pt(2))
    add_para(tf_c, "• Cognitive Burden Reduction: 32.2% lower modeled load due to surprisal knapsack.", font_size=9.5, space_after=Pt(0))


def build_slide_9_comparative(prs):
    slide = prs.slides[8]
    clear_slide_content(slide)
    add_header(slide, "8. COMPARATIVE ANALYSIS: Workload Profiles & Macro Queueing Impact", "COMPARATIVE ANALYSIS")

    # Left: NASA-TLX Radar
    radar_path = Path("latex/IJDS/figures/fig_nasa_tlx_radar.png")
    if radar_path.exists():
        slide.shapes.add_picture(str(radar_path), Inches(0.60), Inches(2.48), Inches(5.20), Inches(3.20))

    # Left Callout Card
    add_card(slide, Inches(0.60), Inches(5.75), Inches(5.20), Inches(1.28))
    tb_l = slide.shapes.add_textbox(Inches(0.75), Inches(5.83), Inches(4.90), Inches(1.10))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0

    add_para(tf_l, "NASA-TLX Multidimensional Burden Reduction:", font_size=11, bold=True, color=NAVY_PRIMARY, space_after=Pt(3))
    add_para(tf_l, "• Mental Demand: 38.3 (SDT) vs. 66.7 (Control) → -42.5% reduction.", font_size=9.5, space_after=Pt(2))
    add_para(tf_l, "• Frustration: 17.5 (SDT) vs. 39.9 (Control) → -56.2% reduction.", font_size=9.5, space_after=Pt(2))
    add_para(tf_l, "• Effort: 38.4 vs. 64.6 (-40.6%)  |  Perceived Performance: 72.8 vs. 56.4 (+29.1%).", font_size=9.5, space_after=Pt(0))

    # Right: Queueing Impact Plot
    queue_path = Path("latex/IJDS/figures/fig_queueing_impact.png")
    if queue_path.exists():
        slide.shapes.add_picture(str(queue_path), Inches(6.15), Inches(2.48), Inches(6.58), Inches(3.20))

    # Right Callout Card
    add_card(slide, Inches(6.15), Inches(5.75), Inches(6.58), Inches(1.28))
    tb_r = slide.shapes.add_textbox(Inches(6.30), Inches(5.83), Inches(6.28), Inches(1.10))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    add_para(tf_r, "Theorem 3: Super-Linear Queue Congestion Relief:", font_size=11, bold=True, color=SUCCESS_DARK, space_after=Pt(3))
    add_para(tf_r, "• Erlang C M/M/c Model: Hospital ED staffed by c=8 MDs, arrival λ=6.0/hr, utilization ρ=93.75%.", font_size=9.5, space_after=Pt(2))
    add_para(tf_r, "• Super-Linear Derivative: ∂Wq/∂(Δt) scales quadratically as O((1-ρ)^-2) near saturation.", font_size=9.5, space_after=Pt(2))
    add_para(tf_r, "• 12.31 Minutes Saved per Patient: Wait time drops from 121.10 to 108.79 min.", font_size=9.5, space_after=Pt(2))
    add_para(tf_r, "• >10,200 Hours of Annual Boarding Eliminated for an urban ED managing 50k patients.", font_size=9.5, space_after=Pt(0))


def build_slide_10_conclusion(prs):
    slide = prs.slides[9]
    clear_slide_content(slide)
    add_header(slide, "9. CONCLUSION & FUTURE WORK: Summary of Findings & Translational Roadmap", "CONCLUSION & HORIZONS")

    card_w = Inches(5.90)
    card_h = Inches(4.55)

    # Left Card: Contributions
    add_card(slide, Inches(0.60), Inches(2.48), card_w, card_h)
    tb_l = slide.shapes.add_textbox(Inches(0.80), Inches(2.68), card_w - Inches(0.40), card_h - Inches(0.40))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0

    add_para(tf_l, "Key Scientific & Operational Contributions", font_size=13, bold=True, color=NAVY_PRIMARY, space_after=Pt(6))
    
    add_para(tf_l, "1. Discrete Cuts vs. Manifold Smoothing:", font_size=11, bold=True, color=NAVY_ACCENT, space_after=Pt(2))
    add_para(tf_l, "Cheeger bisections on hypergraph Laplacians provably eliminate context leakage (P_leak ≤ √(2τ_pivot)), avoiding the exponential smoothing lag of Riemannian manifolds.", font_size=10, space_after=Pt(5))

    add_para(tf_l, "2. Continuous Waveform State-Spaces:", font_size=11, bold=True, color=NAVY_ACCENT, space_after=Pt(2))
    add_para(tf_l, "Unifies continuous 125 Hz telemetry via linear state-space Mamba-2 with discrete EHR text into polyadic hyperedges.", font_size=10, space_after=Pt(5))

    add_para(tf_l, "3. Cognitive-Bounded Active Prefetching:", font_size=11, bold=True, color=NAVY_ACCENT, space_after=Pt(2))
    add_para(tf_l, "Formulates chart staging as a 0/1 knapsack bounded by Sweller surprisal limits (Cmax ≤ 7.0 bits), curbing alert fatigue.", font_size=10, space_after=Pt(5))

    add_para(tf_l, "4. Micro-to-Macro Queueing Leverage:", font_size=11, bold=True, color=NAVY_ACCENT, space_after=Pt(2))
    add_para(tf_l, "Proves super-linear delay leverage O((1-ρ)^-2) in congested EDs, translating 25.6s search gains into 12.3-min wait reductions.", font_size=10, space_after=Pt(0))

    # Right Card: Future Work
    add_card(slide, Inches(6.83), Inches(2.48), card_w, card_h)
    tb_r = slide.shapes.add_textbox(Inches(7.03), Inches(2.68), card_w - Inches(0.40), card_h - Inches(0.40))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    add_para(tf_r, "Translational Roadmap & Future Horizons", font_size=13, bold=True, color=SUCCESS_DARK, space_after=Pt(6))

    add_para(tf_r, "1. SMART-on-FHIR Sidecar Integration:", font_size=11, bold=True, color=SUCCESS_DARK, space_after=Pt(2))
    add_para(tf_r, "Packaging as lightweight microservices connecting to commercial EHRs (Epic/Cerner) via CDS Hooks with sub-15ms Redis caches.", font_size=10, space_after=Pt(5))

    add_para(tf_r, "2. Prospective Silent Shadow Trials:", font_size=11, bold=True, color=SUCCESS_DARK, space_after=Pt(2))
    add_para(tf_r, "Live validation in regional Level-1 Trauma Emergency Departments to evaluate clinician ergonomics without altering care.", font_size=10, space_after=Pt(5))

    add_para(tf_r, "3. Federated Multi-Facility Hypergraphs:", font_size=11, bold=True, color=SUCCESS_DARK, space_after=Pt(2))
    add_para(tf_r, "Extending spectral tracking to regional transfer networks with privacy-preserving federated spectral clustering.", font_size=10, space_after=Pt(5))

    add_para(tf_r, "4. Human-in-the-Loop Safeguards:", font_size=11, bold=True, color=SUCCESS_DARK, space_after=Pt(2))
    add_para(tf_r, "Pruned context branches remain one-click restorable in a sidecar drawer, ensuring full clinician auditability and agency.", font_size=10, space_after=Pt(0))


def build_slide_11_references(prs):
    slide = prs.slides[10]
    clear_slide_content(slide)
    add_header(slide, "10. REFERENCES: Core Literature Citations", "BIBLIOGRAPHIC CITATIONS")

    # 2 Column layout
    col_w = Inches(5.90)
    col_h = Inches(4.55)

    add_card(slide, Inches(0.60), Inches(2.48), col_w, col_h)
    tb_l = slide.shapes.add_textbox(Inches(0.80), Inches(2.65), col_w - Inches(0.40), col_h - Inches(0.35))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0

    refs_col1 = [
        ("Baeza-Yates, R., & Ribeiro-Neto, B. (1999).", "Modern Information Retrieval. ACM Press / Addison-Wesley."),
        ("Chung, F. R. (1997).", "Spectral Graph Theory. CBMS Regional Conference Series in Mathematics, American Mathematical Society, Vol. 92."),
        ("Gu, A., & Dao, T. (2024).", "Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752."),
        ("Hyland, S. L., Faltys, M., Hüser, M., Lyu, X., et al. (2020).", "Early prediction of circulatory failure in the intensive care unit using machine learning. Nature Medicine, 26(3), 364–373."),
        ("Jiang, P., et al. (2024).", "GraphCare: Healthcare predictive modeling with knowledge graphs. In Proceedings of the ACM Web Conference (WWW 2024).")
    ]

    for auth, title in refs_col1:
        add_para(tf_l, auth, font_size=10.5, bold=True, color=NAVY_PRIMARY, space_after=Pt(1))
        add_para(tf_l, title, font_size=9.5, bold=False, color=SLATE_DARK, space_after=Pt(8))

    add_card(slide, Inches(6.83), Inches(2.48), col_w, col_h)
    tb_r = slide.shapes.add_textbox(Inches(7.03), Inches(2.65), col_w - Inches(0.40), col_h - Inches(0.35))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    refs_col2 = [
        ("Jin, Q., Kim, S., Chen, Q., et al. (2023).", "MedCPT: Contrastive pre-trained biomedical dual-encoders for search. Bioinformatics, 39(11), btad651."),
        ("Johnson, A. E., Bulgarelli, L., Shen, L., et al. (2023).", "MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10(1), 1."),
        ("Santhanam, K., Khattab, O., Saad-Falcon, J., et al. (2022).", "ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In NAACL-HLT 2022."),
        ("Sweller, J. (1988).", "Cognitive load during problem solving: Effects on learning. Cognitive Science, 12(2), 257–285."),
        ("Zhou, D., Huang, J., & Schölkopf, B. (2006).", "Learning with hypergraphs: Clustering and classification. In Advances in Neural Information Processing Systems (NeurIPS 2006).")
    ]

    for auth, title in refs_col2:
        add_para(tf_r, auth, font_size=10.5, bold=True, color=NAVY_PRIMARY, space_after=Pt(1))
        add_para(tf_r, title, font_size=9.5, bold=False, color=SLATE_DARK, space_after=Pt(8))


def add_speaker_notes(prs):
    """Add structured oral scripts, 12-minute timeboxed guides, and Q&A cues to slide notes."""
    notes = {
        0: (
            "[TIMING: 0:00 - 0:45 | ~45 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Good morning session chairs and colleagues. My name is Hemanth Manchabale Papachappa from Stanford University, "
            "presenting our work co-authored with Aniruddha Ganguly from Jadavpur University: 'Spectral Diagnostic Trajectories: "
            "Multi-Modal Hypergraph State-Space Tracking, Context Decoupling, and Congestion Mitigation in Stochastic Clinical Service Systems'.\n\n"
            "We address a critical bottleneck at the intersection of AI, clinical workflow, and queueing operations: how clinicians "
            "retrieve information under acute uncertainty, and why eliminating search friction directly unlocks macroscopic throughput "
            "gains in congested hospital emergency departments.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Stanford University & Jadavpur University collaboration.\n"
            "• Core intersection: Healthcare AI, Spectral Graph Theory, and Stochastic Operations Research."
        ),
        1: (
            "[TIMING: 0:45 - 1:15 | ~30 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Here is our roadmap. We begin with clinical search bottlenecks and research gaps in the literature. "
            "We then present the Spectral Diagnostic Trajectories architecture and its theoretical Cheeger bounds. "
            "Finally, we report empirical outcomes across 1,000 standardized vignettes and 9,000 sessions, examine clinician "
            "workload and macro-queueing impacts, and conclude with the translational roadmap.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Logical progression from clinical challenge to mathematical guarantees and hospital-wide operational impact."
        ),
        2: (
            "[TIMING: 1:15 - 2:15 | ~1.0 min]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"To summarize the paper's core contributions: Clinicians in acute care spend over 33% of active shifts querying fragmented EHRs. "
            "When a patient's presentation abruptly pivots—such as an initial pneumonia impression revealing pulmonary embolism—conventional "
            "retrievers fail: stateless models discard workflow context, while continuous Riemannian smoothing lags behind acute bifurcations.\n\n"
            "SDT unifies continuous 125 Hz telemetry and clinical notes into dynamic hypergraphs. By tracking the hypergraph Laplacian "
            "spectral gap in linear time O(|Et|), we detect pivots instantly and apply Cheeger bisections to sever obsolete context.\n\n"
            "This achieves a 28.9% TTCI speedup, 93.9% resolution accuracy, a 42.5% mental workload reduction, and eliminates 12.3 minutes "
            "of emergency waiting room delay per patient.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Problem: Context contamination during acute diagnostic regime shifts.\n"
            "• SDT Solution: Dynamic hypergraph tracking, Cheeger decoupling cut, and cognitive knapsack prefetching.\n"
            "• Key Gains: 28.9% faster insight, 93.9% accuracy, 12.3 min wait time reduction."
        ),
        3: (
            "[TIMING: 2:15 - 3:30 | ~1 min 15 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Let us examine the operational realities. An emergency physician manages 20 to 30 concurrent, undifferentiated patients "
            "inundated with multi-rate data: continuous 125 Hz ECG waveforms, labs, radiographs, and narrative notes.\n\n"
            "Crucially, 53% of complex emergency cases undergo acute diagnostic regime shifts. In current systems, obsolete evidence "
            "remains trapped in the context window. This 'context poisoning' biases downstream retrieval and induces cognitive fatigue.\n\n"
            "Furthermore, emergency departments operate under heavy traffic (rho > 0.90). In this regime, waiting time scales quadratically: "
            "O((1-rho)^-2). Even modest search delays amplify into hours of department-wide boarding. Saving 25 seconds per encounter "
            "is high-leverage because it shifts operations away from catastrophic delay asymptotes.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Information overload: 20-30 concurrent patients, continuous multi-rate data.\n"
            "• Context poisoning: 53% of cases pivot, trapping stale hypotheses in retrieval prompts.\n"
            "• Queueing leverage: Near-saturation queues magnify individual task delays into system-wide gridlock."
        ),
        4: (
            "[TIMING: 3:30 - 4:45 | ~1 min 15 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Evaluating the literature reveals four existing paradigms:\n"
            "1. Lexical search (BM25) is stateless, ignores clinical trajectory, and lacks waveform support.\n"
            "2. Dense Semantic RAG (MedCPT) captures semantic text similarities, but remains memoryless between turns.\n"
            "3. Knowledge Graphs (GraphCare) capture static medical ontologies, but cannot ingest continuous 125 Hz telemetry.\n"
            "4. Continuous Riemannian Manifolds (GDT) model drift on SPD cones, but continuous geodesic smoothing mathematically "
            "lags behind acute discrete bifurcations.\n\n"
            "SDT bridges these worlds: dynamic hypergraph Laplacians, continuous Mamba-2 telemetry state-spaces, and discrete Cheeger cuts "
            "that isolate obsolete context with formal mathematical guarantees.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Highlight the matrix columns.\n"
            "• Emphasize the core contrast: Continuous manifold smoothing vs. discrete Cheeger topological cuts."
        ),
        5: (
            "[TIMING: 4:45 - 5:45 | ~1.0 min]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"This analysis uncovers four critical research gaps:\n"
            "Gap 1: The Memoryless vs. Over-Smoothed Dilemma. Stateless models discard trajectory; continuous manifold models over-smooth.\n"
            "Gap 2: Neglect of High-Frequency Waveforms. Current RAG systems index text narratives and billing codes, ignoring continuous "
            "125 Hz physiological telemetry that carries the earliest markers of acute instability.\n"
            "Gap 3: Unbounded Context Contamination. Systems lack topological mechanisms to prune obsolete context, causing prompt poisoning "
            "and clinician cognitive fatigue.\n"
            "Gap 4: Micro-Latency to Macro-Congestion Disconnect. Prior work benchmarks millisecond query latency in isolation, ignoring "
            "the super-linear queueing physics of congested hospital systems.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Four foundational gaps: memoryless dilemma, waveform omission, prompt pollution, and queueing disconnect."
        ),
        6: (
            "[TIMING: 5:45 - 7:45 | ~2.0 min]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Here is the 5-stage SDT architecture:\n"
            "Stage 1: Multi-Modal Ingestion. Clinical notes are embedded via ClinicalBERT; continuous 125 Hz waveforms are ingested in linear time "
            "via selective state-space Mamba-2 models.\n"
            "Stage 2: Dynamic Hypergraph. Polyadic hyperedges bind diagnostic entities, notes, and telemetry with composite semantic, temporal, and severity weights.\n"
            "Stage 3: Linear-Time Pivot Detection. We track the normalized Laplacian Fiedler value lambda_2 in O(|Et|) time using warm-started Lanczos iteration. "
            "A sudden drop in spectral gap (Delta lambda_2 > 0.15) detects diagnostic regime shifts in under 15 milliseconds.\n"
            "Stage 4: Cheeger Decoupling Cut. We perform a Cheeger bisection on the Fiedler vector to prune obsolete context branches. In Theorem 1, "
            "we prove that context leakage is strictly bounded by P_leak <= sqrt(2 * tau_pivot) = 0.548.\n"
            "Stage 5: Cognitive Prefetching. Staging is formulated as an information-theoretic 0/1 knapsack bounded by Sweller's working memory limit "
            "(Cmax <= 7.0 bits), curbing alert fatigue while feeding into the hospital queueing model.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Step through the 5 visual stages.\n"
            "• Complexity is linear O(|Et|), achieving 112ms end-to-end latency.\n"
            "• Theorem 1 guarantees strict context isolation during acute bifurcations."
        ),
        7: (
            "[TIMING: 7:45 - 9:30 | ~1 min 45 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"We evaluated SDT on a 1,000-vignette benchmark spanning MIMIC-IV, HiRID, and MIMIC-CXR across 9,000 simulated sessions across 9 arms.\n\n"
            "Key empirical outcomes:\n"
            "1. Time-to-Clinical-Insight: SDT achieves 62.97 seconds versus 88.52 seconds for Control—a 28.9% speedup (p < 10^-12, large effect size d = 1.38). "
            "It also beats continuous Riemannian manifolds (GDT) by 8.7% because it avoids smoothing lag.\n"
            "2. Diagnostic Accuracy: Reaches 93.9% resolution rate, outperforming dense RAG (88.4%) and GDT (91.5%).\n"
            "3. Retrieval Quality: Recall@5 reaches 0.914 and MRR reaches 0.862 at 112.9 ms per-step latency.\n"
            "4. Ablations prove that omitting telemetry degrades TTCI to 69.5s; omitting Cheeger cuts degrades TTCI to 71.3s; and omitting surprisal bounds "
            "spikes cognitive load by 39%.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Reference the 9-arm results table and bar chart.\n"
            "• 28.9% TTCI speedup, 93.9% accuracy, 0.914 Recall@5, 112.9ms latency.\n"
            "• Ablation confirms each architectural component is strictly necessary."
        ),
        8: (
            "[TIMING: 9:30 - 11:00 | ~1 min 30 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Next, we examine clinician ergonomics and macro-queueing dynamics.\n\n"
            "On the left: NASA-TLX profiles show dramatic workload reductions. Mental Demand drops from 66.7 to 38.3 (-42.5%), Frustration drops "
            "from 39.9 to 17.5 (-56.2%), and Effort decreases by 40.6% because obsolete context is pruned before polluting the screen.\n\n"
            "On the right: Macro hospital queueing impact in an emergency department staffed by 8 physicians at 93.75% utilization. "
            "In Theorem 3, we showed that delay sensitivity scales as O((1-rho)^-2). Because clinicians spend 33% of their time searching records, "
            "our 25.55-second retrieval speedup reduces consultation time from 18.0 to 17.57 minutes.\n\n"
            "In heavy traffic, this 2.4% service-rate increase cascades into a 12.31-minute reduction in waiting room queue delay per patient, "
            "eliminating over 10,200 hours of cumulative boarding annually for a 50,000-visit hospital.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• NASA-TLX radar: Mental Demand -42.5%, Frustration -56.2%.\n"
            "• Theorem 3 Queueing curve: Super-linear sensitivity O((1-rho)^-2).\n"
            "• Macro payoff: 25.6s search speedup -> 12.31 min wait reduction -> >10,200 annual boarding hours eliminated."
        ),
        9: (
            "[TIMING: 11:00 - 11:45 | ~45 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"To conclude, SDT delivers four scientific breakthroughs: discrete Cheeger cuts over continuous manifold smoothing, "
            "continuous 125 Hz telemetry state-space tracking, cognitive-bounded surprisal prefetching, and formal micro-to-macro queueing leverage.\n\n"
            "On the translational horizon: We are packaging SDT into a SMART-on-FHIR sidecar for Epic and Cerner, preparing prospective "
            "silent shadow trials in Level-1 trauma centers, and developing federated multi-facility hypergraphs with one-click human-in-the-loop "
            "context restoration.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Reiterate the 4 core contributions.\n"
            "• Outline the practical translational roadmap: SMART-on-FHIR, silent shadow trials, federated hypergraphs."
        ),
        10: (
            "[TIMING: 11:45 - 12:00 | ~15 sec]\n"
            "ORAL PRESENTATION SCRIPT:\n"
            "\"Here are our primary references grounding this research. Thank you very much for your time, and I look forward to your questions.\"\n\n"
            "KEY TALKING POINTS:\n"
            "• Acknowledge core literature and open the floor to Q&A.\n"
            "• Quick Q&A cues:\n"
            "  - Eigendecomposition speed: Warm-started Lanczos converges in 5-8 iterations (<15ms).\n"
            "  - Censoring boundary: Average TTCI is ~63s, far below the 300s censoring limit.\n"
            "  - Safety: Pruned context remains restorable in a collapsible drawer."
        )
    }

    for idx, text in notes.items():
        if idx < len(prs.slides):
            slide = prs.slides[idx]
            notes_slide = slide.notes_slide
            tf = notes_slide.notes_text_frame
            tf.text = text


def main():
    ppt_path = Path("latex/IJDS/ICI3T_2026_PPT_Template_Adroid.pptx")
    print(f"Loading template: {ppt_path}")
    prs = pptx.Presentation(str(ppt_path))

    print(f"Template contains {len(prs.slides)} slides.")

    build_slide_1_title(prs)
    print("✓ Slide 1: Title & Authors updated")

    build_slide_2_index(prs)
    print("✓ Slide 2: Index / Agenda updated")

    build_slide_3_abstract(prs)
    print("✓ Slide 3: Abstract updated")

    build_slide_4_introduction(prs)
    print("✓ Slide 4: Introduction & Operational Bottlenecks updated")

    build_slide_5_literature(prs)
    print("✓ Slide 5: Literature Review Table updated")

    build_slide_6_gaps(prs)
    print("✓ Slide 6: Research Gaps updated")

    build_slide_7_methodology(prs)
    print("✓ Slide 7: Proposed Methodology Flowchart updated")

    build_slide_8_results(prs)
    print("✓ Slide 8: Results & Discussion Table + Bar Chart updated")

    build_slide_9_comparative(prs)
    print("✓ Slide 9: Comparative Analysis (NASA-TLX & Queueing) updated")

    build_slide_10_conclusion(prs)
    print("✓ Slide 10: Conclusion & Future Work updated")

    build_slide_11_references(prs)
    print("✓ Slide 11: References updated")

    add_speaker_notes(prs)
    print("✓ Speaker notes attached to all 11 slides")

    prs.save(str(ppt_path))
    print(f"\n🎉 Successfully saved updated presentation with speaker notes to {ppt_path}")


if __name__ == "__main__":
    main()
