# Theoretical Framework: Quiz Vulnerability Assessment

This document provides the scholarly and theoretical foundations underpinning the Quiz Vulnerability Assessment Framework (QVAF). It is intended for educators, educational designers, and researchers who wish to understand the conceptual basis for the tool's design decisions.

---

## Table of Contents

1. [The Problem: AI Vulnerability in Online Quizzes](#1-the-problem-ai-vulnerability-in-online-quizzes)
2. [Regulatory Context: TEQSA 2025 Guidance](#2-regulatory-context-teqsa-2025-guidance)
3. [Theoretical Foundation: Why Redesign Beats Detection](#3-theoretical-foundation-why-redesign-beats-detection)
4. [The Cognitive Demand Taxonomy](#4-the-cognitive-demand-taxonomy)
5. [Empirical Basis: LLM Performance Patterns](#5-empirical-basis-llm-performance-patterns)
6. [Methodology: How the Framework Assesses Vulnerability](#6-methodology-how-the-framework-assesses-vulnerability)
7. [Design Philosophy: Decision-Support, Not Prescription](#7-design-philosophy-decision-support-not-prescription)
8. [Scope and Limitations](#8-scope-and-limitations)
9. [References](#9-references)

---

## 1. The Problem: AI Vulnerability in Online Quizzes

Online quizzes—particularly those using multiple-choice questions (MCQs), true/false items, and short-answer formats—face a specific vulnerability in the age of generative AI. Students can copy quiz questions directly into an AI system (such as ChatGPT, Claude, or a locally-run LLM) and receive answers that may be correct, plausible-sounding, or confidently stated regardless of accuracy.

This behaviour is:

- **Technologically straightforward**: No special skills required; copy-paste into any chat interface
- **Difficult to detect**: AI-generated text detection tools remain unreliable, with false positive rates that make enforcement problematic (Weber-Wulff et al., 2023)
- **Potentially widespread**: Survey evidence suggests significant proportions of students have used or would consider using AI assistance on assessments (Smolansky et al., 2024)

The challenge is not that AI *always* answers correctly—empirical evidence shows significant variability depending on question type, domain, and model (Newton et al., 2024). The challenge is that educators often lack visibility into *which* of their questions are vulnerable and *why*.

### What This Tool Addresses

QVAF addresses one specific cheating behaviour: **students copying quiz questions directly into an AI system and using the AI-generated answer**.

This scope is deliberately narrow. By focusing on a tractable, well-defined problem, the tool can provide actionable insights rather than vague warnings about "AI risks."

### What This Tool Does Not Address

The following behaviours are explicitly outside scope:

| Behaviour | Why Excluded |
|-----------|--------------|
| Contract cheating (human assistance) | Different intervention required; not an AI problem |
| Sophisticated prompt engineering | Assumes basic copy-paste behaviour most students would use |
| Real-time AI assistance during proctored exams | Requires different controls (proctoring, browser lockdown) |
| AI-assisted learning followed by legitimate recall | This is arguably desirable learning behaviour |
| Multi-modal cheating (photo → AI) | Addressed partially via image-based question testing |

Documenting these exclusions enables appropriate interpretation of results.

---

## 2. Regulatory Context: TEQSA 2025 Guidance

The Quiz Vulnerability Assessment Framework aligns with guidance from Australia's Tertiary Education Quality and Standards Agency (TEQSA), which has published two foundational documents on assessment reform in response to generative AI.

### Assessment Reform for the Age of Artificial Intelligence (2023)

Lodge, Howard, Bearman, Dawson, and Associates (2023) established two guiding principles:

> **Principle 1**: "Assessment and learning experiences equip students to participate ethically and actively in a society where AI is ubiquitous."

> **Principle 2**: "Forming trustworthy judgements about student learning in a time of AI requires multiple, inclusive and contextualised approaches to assessment."

The document articulates five propositions for assessment reform, including emphasis on:

- Appropriate, authentic engagement with AI
- Systemic approaches to program-level assessment
- Assessment of learning processes, not just products
- Security at meaningful points across programs

### Enacting Assessment Reform (2025)

The follow-up document (Lodge et al., 2025) translates principles into three strategic pathways:

1. **Program-wide assessment reform**: Treating assessment as a connected system across entire degrees
2. **Unit-level assurance**: Embedding at least one secure assessment within every unit
3. **Hybrid approaches**: Combining program-wide and unit-level strategies

Critically, the 2025 guidance states:

> "Rather than investing primarily in detection mechanisms, institutions need to emphasise the redesign of assessment to capture authentic demonstrations of student capability and comprehension."

### QVAF's Alignment

QVAF supports the TEQSA framework by enabling educators to:

- **Identify vulnerable questions proactively** before quiz deployment
- **Understand patterns** in AI performance across their assessment items
- **Make informed decisions** about which questions to redesign, replace, or retain
- **Focus resources** on questions where redesign will have greatest impact

The tool embodies the guidance principle of designing assessments where "gen AI use is irrelevant to the demonstration of learning outcomes"—not by banning AI, but by identifying questions where AI assistance provides unfair advantage and supporting informed redesign.

---

## 3. Theoretical Foundation: Why Redesign Beats Detection

### The Structural vs. Discursive Distinction

Corbin, Dawson, and Liu (2025) provide the theoretical framework distinguishing between **discursive changes** and **structural changes** to assessment:

| Change Type | Examples | Limitations |
|-------------|----------|-------------|
| **Discursive** | AI use policies, honour codes, permitted/prohibited declarations, acknowledgment statements | "They say much but change little. They direct behaviour they cannot monitor. They prohibit actions they cannot detect." |
| **Structural** | Redesigning question types, adding oral components, changing assessment timing, modifying what is assessed | Fundamentally alters the mechanics of assessment; does not rely on student compliance |

QVAF supports structural change by providing the diagnostic information educators need to make informed redesign decisions.

### Why Detection Is Insufficient

AI-generated text detection faces fundamental limitations:

1. **Accuracy problems**: Detection tools show unacceptably high false positive rates, particularly for non-native English speakers (Liang et al., 2023)
2. **Arms race dynamics**: As detection improves, generation techniques evolve to evade detection
3. **Paraphrasing defeats detection**: Simple rewording of AI output renders most detection ineffective
4. **Burden of proof**: Academic integrity processes require evidence beyond reasonable doubt; probabilistic detection rarely meets this standard

The AI Assessment Scale (Perkins et al., 2024), adopted by over 250 institutions worldwide, explicitly acknowledges that "permitting any use of AI effectively permits all use of AI" due to undetectability. This recognition shifts focus from enforcement to design.

### The Case for Proactive Redesign

Proactive assessment redesign offers several advantages:

- **Scalable**: One redesign effort protects all future cohorts
- **Fair**: Does not rely on catching individual students
- **Educational**: Can improve assessment quality independent of AI concerns
- **Sustainable**: Does not require ongoing detection infrastructure

QVAF operationalises this approach by making vulnerability visible and actionable.

---

## 4. The Cognitive Demand Taxonomy

### Theoretical Foundations

The QVAF cognitive demand taxonomy synthesises three established educational frameworks:

#### Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001)

Six cognitive process levels:
- **Remember**: Retrieving knowledge from long-term memory
- **Understand**: Constructing meaning from instruction
- **Apply**: Using procedures in given situations
- **Analyze**: Breaking material into parts, detecting relationships
- **Evaluate**: Making judgments based on criteria
- **Create**: Putting elements together to form a new whole

#### Webb's Depth of Knowledge (Webb, 1997)

Four levels of cognitive complexity:
- **DOK 1 (Recall)**: Recall of facts, definitions, procedures
- **DOK 2 (Skill/Concept)**: Use of information, mental processing
- **DOK 3 (Strategic Thinking)**: Reasoning, planning, evidence-based justification
- **DOK 4 (Extended Thinking)**: Complex reasoning, synthesis over time

Key distinction: DOK focuses on the *depth* of understanding required, not just the type of activity. A question asking students to "analyze" data might be DOK 2 if the analysis is routine, or DOK 3 if it requires non-routine judgment.

#### SOLO Taxonomy (Biggs & Collis, 1982)

Describes levels of understanding demonstrated:
- **Prestructural**: Missing the point
- **Unistructural**: One relevant aspect
- **Multistructural**: Several relevant aspects (not integrated)
- **Relational**: Integrated understanding
- **Extended Abstract**: Generalisation to new domain

### The QVAF Five-Level Taxonomy

Synthesising these frameworks for application to quiz vulnerability assessment:

| Level | Name | Description | Typical Indicators |
|-------|------|-------------|-------------------|
| **1** | Recall | Direct retrieval of memorised facts, definitions, or procedures | "Define...", "What is...", "List...", "Name..." |
| **2** | Routine Application | Applying known procedures where method selection is obvious | "Calculate using...", "Apply the formula...", "Follow the steps..." |
| **3** | Conceptual Understanding | Demonstrating understanding of relationships between concepts | "Explain why...", "How does X relate to Y...", "Compare..." |
| **4** | Analytical Reasoning | Breaking down complex information; evaluating evidence | "Analyze the data...", "Evaluate the argument...", "What conclusions..." |
| **5** | Strategic Integration | Synthesising multiple sources; applying to novel situations | "Design a solution...", "How would you address this new scenario..." |

### Discipline-Agnostic Application

The taxonomy is designed for application across disciplines. Level indicators should be interpreted in context:

| Discipline | Level 1 Example | Level 5 Example |
|------------|-----------------|-----------------|
| Psychology | "Define operant conditioning" | "Design an intervention for a novel behavioural problem using multiple theoretical frameworks" |
| Engineering | "State the formula for stress" | "Propose a design solution for a novel constraint set requiring trade-off analysis" |
| Business | "List the 4 Ps of marketing" | "Develop a market entry strategy for an unfamiliar market given incomplete information" |
| Nursing | "Name the stages of wound healing" | "Prioritise care for multiple patients with conflicting needs and resource constraints" |

---

## 5. Empirical Basis: LLM Performance Patterns

### The Cognitive Complexity Gradient

Empirical research demonstrates that LLMs perform significantly better on lower cognitive levels and struggle with higher-order thinking. This pattern provides the scientific basis for using cognitive classification as a vulnerability indicator.

Newton, Da Silva, and Berry (2024) conducted the largest scoping review to date: **53 studies, 114 question sets, 49,014 MCQs**. Key findings:

- GPT-3/3.5: Performed better than random guessing but failed most examinations
- GPT-4: Passed most examinations with performance on par with human subjects
- **Cognitive domain effect**: Statistically significant better performance on Remember/Understand questions compared to Apply/Analyze questions (P=0.041 for GPT-3.5, P=0.003 for GPT-4, P=0.017 for Bard)

Huber and Niklaus (2025) directly mapped LLM benchmark performance to Bloom's Taxonomy:

> "LLMs generally perform better on the lower end of Bloom's Taxonomy."

Testing GPT-4, GPT-4o, Claude 3, and Llama 3, they documented consistent performance degradation as cognitive complexity increases.

### Question Format Effects

Format matters independently of cognitive level:

- **Image-based questions**: Nguyen et al. (2025) found accuracy dropped from 74.8-86.2% (text MCQs) to 61.7-63.8% (image-based)—a **20+ percentage point reduction** simply from requiring visual interpretation
- **Open-style questions**: Li et al. (2024) demonstrated ~25% lower accuracy compared to MCQs across all models tested

### Confidence Calibration Problems

LLMs are systematically overconfident when wrong:

- Xiong et al. (2024): Verbalized confidence values cluster in the 80-100% range regardless of actual accuracy
- This creates a specific vulnerability pattern: **high confidence + incorrect answer** indicates the question successfully exploits LLM limitations

### RAG Effects

Retrieval-Augmented Generation (RAG)—providing LLMs with course materials—affects performance:

- Can improve accuracy on domain-specific factual questions
- Does not reliably improve performance on analytical or novel application questions
- Magesh et al. (2024) found even sophisticated RAG systems hallucinate 17-33% of responses

QVAF tests both conditions (with and without course materials) to identify questions where RAG access specifically increases vulnerability.

---

## 6. Methodology: How the Framework Assesses Vulnerability

### Two-Condition Testing

QVAF tests each question under two conditions:

| Condition | What It Simulates | What It Reveals |
|-----------|-------------------|-----------------|
| **Baseline** (no RAG) | Student copying question into general-purpose AI | Vulnerability to general AI knowledge |
| **Enhanced** (with RAG) | Student using AI with access to course materials | Additional vulnerability from course-specific content |

### Metrics Collected

For each question, the framework collects:

1. **AI Correctness**: Did the AI answer correctly? (Yes/No)
2. **AI Confidence**: How confident was the AI in its answer? (0-100%)
3. **Answer Consistency**: When tested multiple times, how often did the AI give the same answer? (e.g., "8/10")

### Correctness Patterns

The combination of baseline and RAG results creates descriptive patterns:

| Pattern | Baseline | RAG | Interpretation |
|---------|----------|-----|----------------|
| **Correct Both** | ✓ | ✓ | AI succeeds regardless of materials access |
| **Correct RAG Only** | ✗ | ✓ | Course materials specifically enable AI success |
| **Incorrect Both** | ✗ | ✗ | Question resists AI assistance |
| **Correct Baseline Only** | ✓ | ✗ | Anomalous; RAG may introduce confusion |

**Important**: These are **descriptive patterns**, not vulnerability judgments. The educator interprets what these patterns mean in their specific context.

### Why Multi-Sample Testing

Even at temperature=0 (greedy decoding), LLMs produce variable outputs due to:

- Floating-point operation ordering
- GPU parallelism effects  
- Batching variations

Multi-sample testing (running each question multiple times) captures meaningful variability:

- A question answered correctly 10/10 times differs meaningfully from one answered correctly 7/10 times
- The latter shows exploitable uncertainty that well-constructed distractors might leverage

---

## 7. Design Philosophy: Decision-Support, Not Prescription

### The Role of the Subject Matter Expert

QVAF is designed as an **assistant to educators**, not as an authoritative system:

| The Tool... | The Educator... |
|-------------|-----------------|
| Provides objective metrics | Interprets what metrics mean in context |
| Classifies cognitive demand | Validates or overrides classifications |
| Generates recommendations | Accepts, modifies, or rejects suggestions |
| Identifies patterns | Decides what action to take |

### Why No Categorical Vulnerability Labels

The framework explicitly avoids assigning categorical labels (e.g., "HIGH RISK", "LOW RISK") because:

1. **Context matters**: A question where AI achieves 70% accuracy may be acceptable in a formative quiz but concerning in a high-stakes exam
2. **Stakes vary**: Different assessments warrant different thresholds
3. **Professional autonomy**: Educators should make assessment decisions, not automated systems
4. **Disciplinary variation**: What constitutes "vulnerable" differs by field

### Asset-Based Framing

Research on constructive feedback (Fong et al., 2016) indicates that criticism is perceived as constructive only when it identifies gaps **AND** provides specific directions for improvement.

QVAF reports are designed to:

1. **Lead with what works**: Identify questions that successfully resist AI assistance
2. **Provide context**: Show patterns across the quiz, not just problem areas
3. **Offer actionable paths**: Every identified vulnerability includes potential mitigation strategies
4. **Avoid overwhelm**: Present information progressively, with summary views before detailed analysis

### Risk Mitigation, Not Elimination

QVAF does not promise "AI-proof" assessments because:

1. **No such thing exists**: LLM capabilities evolve continuously
2. **Perfect security is impossible**: Even proctored exams have vulnerabilities
3. **Diminishing returns**: Beyond a certain point, security measures harm legitimate students more than they deter cheaters

The appropriate framing is **risk mitigation**—reducing the effectiveness of the most common AI-assisted cheating behaviour to a level the educator deems acceptable for their context.

---

## 8. Scope and Limitations

### Explicit Scope

QVAF is designed for:

- **Unproctored online quizzes**: Where students have unrestricted access to external resources
- **MCQ, true/false, and short-answer formats**: Auto-gradeable question types
- **Single LLM testing**: Using a representative model (e.g., llama3:8b) rather than comprehensive multi-model benchmarking

### Known Limitations

| Limitation | Implication | Mitigation |
|------------|-------------|------------|
| Single model testing | Results reflect one AI system, not all possible systems | Use a capable, representative model; acknowledge limitation in reports |
| LLM-based classification | Cognitive demand classification uses the same technology being tested | Treat classifications as suggestions requiring educator validation |
| Point-in-time assessment | LLM capabilities change; today's resistant question may be vulnerable tomorrow | Periodic re-testing recommended; framework supports re-assessment |
| MCQ format constraints | Some learning objectives cannot be validly assessed via MCQ regardless of AI-resistance | Tool can identify such cases; "consider alternative format" is a valid recommendation |

### What QVAF Cannot Do

- **Guarantee AI-proof questions**: No tool can make this guarantee
- **Replace educator judgment**: The tool informs decisions; it does not make them
- **Detect AI use after the fact**: This is a proactive redesign tool, not a detection tool
- **Address all cheating behaviours**: Scope is limited to one specific, common behaviour

---

## 9. References

Anderson, L. W., & Krathwohl, D. R. (Eds.). (2001). *A taxonomy for learning, teaching, and assessing: A revision of Bloom's taxonomy of educational objectives* (Complete ed.). Longman.

Biggs, J. B., & Collis, K. F. (1982). *Evaluating the quality of learning: The SOLO taxonomy (Structure of the Observed Learning Outcome)*. Academic Press.

Corbin, T., Dawson, P., & Liu, D. (2025). Talk is cheap: Why structural assessment changes are needed for a time of GenAI. *Assessment & Evaluation in Higher Education*, (online first). https://doi.org/10.1080/02602938.2025.2454314

Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, *13*(3), 319–340. https://doi.org/10.2307/249008

Fong, C. J., Warner, J. R., Williams, K. M., Schallert, D. L., Chen, L. H., Williamson, Z. H., & Lin, S. (2016). Deconstructing constructive criticism: The nature of academic emotions associated with constructive, positive, and negative feedback. *Learning and Individual Differences*, *49*, 393–399. https://doi.org/10.1016/j.lindif.2016.05.019

Huber, M., & Niklaus, J. (2025). LLMs meet Bloom's taxonomy: A cognitive view on large language model evaluations. In *Proceedings of the 31st International Conference on Computational Linguistics (COLING 2025)* (pp. 5234–5251). Association for Computational Linguistics.

Li, Y., Chen, S., & Zhang, Q. (2024). Open-LLM-Leaderboard: From multi-choice to open-style questions for LLMs evaluation, benchmark, and arena. In *Proceedings of LREC-COLING 2024* (pp. 8765–8779). ELRA and ICCL.

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns*, *4*(7), 100779. https://doi.org/10.1016/j.patter.2023.100779

Lodge, J. M., Bearman, M., Dawson, P., Gniel, H., Harper, R., Liu, D., McLean, J., Ucnik, L., & Associates. (2025). *Enacting assessment reform in a time of artificial intelligence*. Tertiary Education Quality and Standards Agency, Australian Government.

Lodge, J. M., Howard, S., Bearman, M., Dawson, P., & Associates. (2023). *Assessment reform for the age of artificial intelligence*. Tertiary Education Quality and Standards Agency, Australian Government.

Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2024). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Stanford Law School Working Paper*.

Newton, P., Da Silva, A., & Berry, S. (2024). ChatGPT performance on multiple choice question examinations in higher education: A pragmatic scoping review. *Assessment & Evaluation in Higher Education*, *49*(6), 781–798. https://doi.org/10.1080/02602938.2023.2299059

Nguyen, T., Tran, H., Le, M., & Pham, K. (2025). Accuracy of latest large language models in answering multiple choice questions in dentistry: A comparative study. *PLOS ONE*, *20*(1), e0317423. https://doi.org/10.1371/journal.pone.0317423

Perkins, M., Furze, L., Roe, J., & MacVaugh, J. (2024). The Artificial Intelligence Assessment Scale (AIAS): A framework for ethical integration of generative AI in educational assessment. *Journal of University Teaching and Learning Practice*, *21*(6). https://doi.org/10.53761/1.21.6.02

Scherer, R., Siddiq, F., & Tondeur, J. (2019). The technology acceptance model (TAM): A meta-analytic structural equation modeling approach to explaining teachers' adoption of digital technology in education. *Computers & Education*, *128*, 13–35. https://doi.org/10.1016/j.compedu.2018.09.009

Smolansky, A., Cram, A., Raduescu, C., Zeide, E., Kovanovic, V., & Joksimovic, S. (2024). Perceived impact of generative AI on assessments: Comparing educator and student perspectives in Australia, Cyprus, and the United States. *Computers and Education Open*, *6*, 100198. https://doi.org/10.1016/j.caeo.2024.100198

Webb, N. L. (1997). *Criteria for alignment of expectations and assessments in mathematics and science education* (Research Monograph No. 6). National Institute for Science Education, University of Wisconsin-Madison.

Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltýnek, T., Guerrero-Dib, J., Popoola, O., Šigut, P., & Waddington, L. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity*, *19*(1), 26. https://doi.org/10.1007/s40979-023-00146-z

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In *Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)*.

---

*This document is part of the Quiz Vulnerability Assessment Framework (QVAF) documentation. For installation and usage instructions, see [README.md](../README.md). For technical architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).*
